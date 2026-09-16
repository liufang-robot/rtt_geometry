// SPDX-License-Identifier: LGPL-2.1-or-later
#pragma once
#include "eigen_type_info.hpp"
#include <rtt/types/Operators.hpp>
#include <rtt/types/OperatorTypes.hpp>
#include <rtt/types/TemplateConstructor.hpp>
#include <rtt/internal/GlobalService.hpp>
#include <cmath>
#include <functional>
#include <memory>

namespace Eigen::rtt {
using RTT::types::newConstructor;
using RTT::types::newBinaryOperator;
using RTT::types::newUnaryOperator;

template<class R, class A, class B> auto binary(const char* op, R (*function)(A, B)) {
    return newBinaryOperator(op, std::function<R(A, B)>(function));
}
template<class R, class A> auto unary(const char* op, R (*function)(A)) {
    return newUnaryOperator(op, std::function<R(A)>(function));
}

// newConstructor(f, false) only warns on implicit conversion in RTT. A separate
// wrapper is necessary to keep fixed/dynamic conversions explicit and prevent
// overloaded arithmetic from silently changing dimensions or return types.
template<class From, bool Automatic = false> class ExactConstructor : public RTT::types::TypeConstructor {
    std::unique_ptr<RTT::types::TypeConstructor> constructor;
public:
    explicit ExactConstructor(RTT::types::TypeConstructor* value) : constructor(value) {}
    Source build(const std::vector<Source>& args) const override {
        if (args.size() != 1 || args[0]->getTypeInfo() != RTT::types::Types()->getTypeInfo<From>())
            return nullptr;
        return constructor->build(args);
    }
    Source convert(Source argument) const override {
        if constexpr (Automatic) return build({argument});
        else return nullptr;
    }
};

template<class T> TypeInfo* type() { return RTT::types::Types()->getTypeInfo<T>(); }
template<class To, class From> To convert(const From& value) {
    checkShape<To>(value.rows(), value.cols());
    return To(value);
}
template<class To, class From> void addConversion() {
    type<To>()->addConstructor(new ExactConstructor<From>(newConstructor(&convert<To, From>)));
}
template<class T> RTT::Service::shared_ptr service() {
    const std::string name = type<T>()->getTypeName().substr(7); // /Eigen/
    return RTT::internal::GlobalService::Instance()->provides("Eigen")->provides(name);
}

template<class T> T zero() { return initialValue<T>(); }
template<class T> T constant(double value) { return T::Constant(value); }
template<class T> T vectorConstant(int size, double value) {
    checkShape<T>(size, 1);
    return T::Constant(size, value);
}
template<class T> T vectorZero(int size) { return vectorConstant<T>(size, 0.0); }
template<class T> T vectorArray(std::vector<double> values) {
    if (values.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Eigen vector is too large");
    const int size = static_cast<int>(values.size());
    checkShape<T>(size, 1);
    return T::Map(values.data(), size);
}
template<class T> T matrixConstant(int rows, int cols, double value) {
    checkShape<T>(rows, cols);
    return T::Constant(rows, cols, value);
}
template<class T> T matrixZero(int rows, int cols) { return matrixConstant<T>(rows, cols, 0.0); }
template<class T> T matrixArray(int rows, int cols, std::vector<double> values) {
    checkShape<T>(rows, cols);
    if (values.size() != static_cast<size_t>(rows) * cols)
        throw std::invalid_argument("Eigen matrix needs rows * cols values in row-major order");
    T result(rows, cols);
    for (int row = 0; row < rows; ++row)
        for (int col = 0; col < cols; ++col) result(row, col) = values[row * cols + col];
    return result;
}
template<class T> T fixedMatrixArray(std::vector<double> values) {
    return matrixArray<T>(T::RowsAtCompileTime, T::ColsAtCompileTime, values);
}
template<class T> T identity() { return T::Identity(); }
template<class T> T matrixIdentity(int rows, int cols) {
    checkShape<T>(rows, cols);
    return T::Identity(rows, cols);
}

template<class T> void sameShape(const T& a, const T& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols())
        throw std::invalid_argument("Eigen operands must have the same dimensions");
}
template<class T> T add(const T& a, const T& b) { sameShape(a, b); return T(a + b); }
template<class T> T subtract(const T& a, const T& b) { sameShape(a, b); return T(a - b); }
template<class T> T negate(const T& value) { return T(-value); }
template<class T> T multiplyScalar(const T& value, double scale) { return T(value * scale); }
template<class T, class Scalar> T scalarMultiply(Scalar scale, const T& value) { return T(value * scale); }
template<class T> T divideScalar(const T& value, double scale) {
    if (!std::isfinite(scale) || scale == 0)
        throw std::domain_error("Eigen divisor must be finite and nonzero");
    return T(value / scale);
}
template<class T> T transpose(const T& value) { return T(value.transpose()); }
template<class T> double norm(const T& value) { return value.stableNorm(); }
template<class T> T normalized(const T& value) {
    const double length = norm(value);
    if (!std::isfinite(length) || length == 0)
        throw std::domain_error("Eigen normalization needs a finite, nonzero vector");
    return T(value / length);
}
template<class T> double dot(const T& a, const T& b) { sameShape(a, b); return a.dot(b); }
inline Vector3d cross(const Vector3d& a, const Vector3d& b) { return a.cross(b); }
template<class T> T multiplyMatrix(const T& a, const T& b) {
    if (a.cols() != b.rows())
        throw std::invalid_argument("Eigen matrix multiplication needs lhs.cols == rhs.rows");
    checkShape<T>(a.rows(), b.cols());
    return T(a * b);
}
template<class M, class V> V multiplyVector(const M& matrix, const V& vector) {
    if (matrix.cols() != vector.size())
        throw std::invalid_argument("Eigen matrix-vector multiplication needs matrix.cols == vector.size");
    return V(matrix * vector);
}
template<class T> void addDenseOperators() {
    auto operators = RTT::types::OperatorRepository::Instance();
    operators->add(binary("+", &add<T>));
    operators->add(binary("-", &subtract<T>));
    operators->add(unary("-", &negate<T>));
    operators->add(binary("*", &multiplyScalar<T>));
    operators->add(binary("*", &scalarMultiply<T, double>));
    operators->add(binary("*", &scalarMultiply<T, int>));
    operators->add(binary("/", &divideScalar<T>));
}

void loadVectorTypes();
void loadMatrixTypes();
void loadQuaternionType();
void loadVectorConstructors();
void loadMatrixConstructors();
void loadQuaternionConstructors();
void loadVectorOperators();
void loadMatrixOperators();
void loadQuaternionOperators();
void loadVectorGlobals();
void loadMatrixGlobals();
void loadQuaternionGlobals();
} // namespace Eigen::rtt
