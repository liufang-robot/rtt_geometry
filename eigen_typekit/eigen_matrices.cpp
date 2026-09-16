// SPDX-License-Identifier: LGPL-2.1-or-later
#include "eigen_scripting.hpp"
#include <sstream>

namespace Eigen::rtt {
template<class T> std::string matrixToString(const T& value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
}
void loadMatrixTypes() {
    auto types = RTT::types::Types();
    types->addType(new MatrixTypeInfo<MatrixXd>("/Eigen/MatrixXd"));
    types->addType(new MatrixTypeInfo<Matrix2d>("/Eigen/Matrix2d"));
    types->addType(new MatrixTypeInfo<Matrix3d>("/Eigen/Matrix3d"));
    types->addType(new MatrixTypeInfo<Matrix4d>("/Eigen/Matrix4d"));
}
template<class T> void matrixConstructors() {
    type<T>()->addConstructor(newConstructor(&matrixZero<T>));
    type<T>()->addConstructor(newConstructor(&matrixConstant<T>));
    type<T>()->addConstructor(newConstructor(&matrixArray<T>));
    if constexpr (T::SizeAtCompileTime != Dynamic) {
        type<T>()->addConstructor(new ExactConstructor<std::vector<double>>(newConstructor(&fixedMatrixArray<T>)));
        addConversion<T, MatrixXd>();
        addConversion<MatrixXd, T>();
    }
}
void loadMatrixConstructors() {
    matrixConstructors<MatrixXd>();
    matrixConstructors<Matrix2d>();
    matrixConstructors<Matrix3d>();
    matrixConstructors<Matrix4d>();
}
template<class M, class V> void matrixOperators() {
    addDenseOperators<M>();
    auto operators = RTT::types::OperatorRepository::Instance();
    operators->add(binary("*", &multiplyMatrix<M>));
    operators->add(binary("*", &multiplyVector<M, V>));
}
void loadMatrixOperators() {
    matrixOperators<MatrixXd, VectorXd>();
    matrixOperators<Matrix2d, Vector2d>();
    matrixOperators<Matrix3d, Vector3d>();
    matrixOperators<Matrix4d, Vector4d>();
}
template<class T> void matrixGlobals() {
    auto api = service<T>();
    if constexpr (T::SizeAtCompileTime == Dynamic) {
        api->addOperation("zero", &matrixZero<T>, RTT::ClientThread);
        api->addOperation("constant", &matrixConstant<T>, RTT::ClientThread);
        api->addOperation("identity", &matrixIdentity<T>, RTT::ClientThread);
    } else {
        api->addOperation("zero", &zero<T>, RTT::ClientThread);
        api->addOperation("constant", &constant<T>, RTT::ClientThread);
        api->addOperation("identity", &identity<T>, RTT::ClientThread);
    }
    api->addOperation("transpose", &transpose<T>, RTT::ClientThread);
    api->addOperation("toString", &matrixToString<T>, RTT::ClientThread)
        .doc("Format all matrix coefficients as rows without TaskBrowser array truncation.");
}
void loadMatrixGlobals() {
    matrixGlobals<MatrixXd>();
    matrixGlobals<Matrix2d>();
    matrixGlobals<Matrix3d>();
    matrixGlobals<Matrix4d>();
}
} // namespace Eigen::rtt
