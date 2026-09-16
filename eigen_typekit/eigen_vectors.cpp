// SPDX-License-Identifier: LGPL-2.1-or-later
#include "eigen_scripting.hpp"

namespace Eigen::rtt {
void loadVectorTypes() {
    auto types = RTT::types::Types();
    types->addType(new VectorTypeInfo<VectorXd>("/Eigen/VectorXd"));
    types->addType(new VectorTypeInfo<Vector2d>("/Eigen/Vector2d"));
    types->addType(new VectorTypeInfo<Vector3d>("/Eigen/Vector3d"));
    types->addType(new VectorTypeInfo<Vector4d>("/Eigen/Vector4d"));
    types->addType(new VectorTypeInfo<Vector6d>("/Eigen/Vector6d"));
}
template<class T> void constructors() {
    type<T>()->addConstructor(new ExactConstructor<int>(newConstructor(&vectorZero<T>)));
    type<T>()->addConstructor(new ExactConstructor<std::vector<double>, true>(newConstructor(&vectorArray<T>)));
    type<T>()->addConstructor(newConstructor(&vectorConstant<T>));
    if constexpr (!std::is_same_v<T, VectorXd>) {
        addConversion<T, VectorXd>();
        addConversion<VectorXd, T>();
    }
}
void loadVectorConstructors() {
    constructors<VectorXd>();
    constructors<Vector2d>();
    constructors<Vector3d>();
    constructors<Vector4d>();
    constructors<Vector6d>();
}
void loadVectorOperators() {
    addDenseOperators<VectorXd>();
    addDenseOperators<Vector2d>();
    addDenseOperators<Vector3d>();
    addDenseOperators<Vector4d>();
    addDenseOperators<Vector6d>();
}
template<class T> void globals() {
    auto api = service<T>();
    if constexpr (T::SizeAtCompileTime == Dynamic) {
        api->addOperation("zero", &vectorZero<T>, RTT::ClientThread);
        api->addOperation("constant", &vectorConstant<T>, RTT::ClientThread);
    } else {
        api->addOperation("zero", &zero<T>, RTT::ClientThread);
        api->addOperation("constant", &constant<T>, RTT::ClientThread);
    }
    api->addOperation("norm", &norm<T>, RTT::ClientThread);
    api->addOperation("normalized", &normalized<T>, RTT::ClientThread);
    api->addOperation("dot", &dot<T>, RTT::ClientThread);
}
void loadVectorGlobals() {
    globals<VectorXd>();
    globals<Vector2d>();
    globals<Vector3d>();
    globals<Vector4d>();
    globals<Vector6d>();
    service<Vector3d>()->addOperation("cross", &cross, RTT::ClientThread);
}
} // namespace Eigen::rtt
