// SPDX-License-Identifier: LGPL-2.1-or-later
#include "eigen_scripting.hpp"

namespace Eigen::rtt {
namespace {
void finiteQuaternion(const Quaterniond& value) {
    if (!value.coeffs().allFinite()) throw std::domain_error("Eigen quaternion coefficients must be finite");
}
Quaterniond quaternion(double w, double x, double y, double z) {
    Quaterniond result(w, x, y, z);
    finiteQuaternion(result);
    return result;
}
Quaterniond quaternionArray(std::vector<double> values) {
    if (values.size() != 4) throw std::invalid_argument("Eigen Quaterniond needs four values: w, x, y, z");
    return quaternion(values[0], values[1], values[2], values[3]);
}
double quaternionNorm(const Quaterniond& value) { return value.coeffs().stableNorm(); }
Quaterniond quaternionNormalized(const Quaterniond& value) {
    finiteQuaternion(value);
    const double scale = value.coeffs().cwiseAbs().maxCoeff();
    if (scale == 0) throw std::domain_error("Eigen quaternion normalization needs a nonzero quaternion");
    Quaterniond result;
    result.coeffs() = value.coeffs() / scale;
    result.coeffs() /= result.coeffs().norm();
    return result;
}
Quaterniond quaternionConjugate(const Quaterniond& value) {
    finiteQuaternion(value);
    return value.conjugate();
}
Quaterniond quaternionInverse(const Quaterniond& value) {
    finiteQuaternion(value);
    const double scale = value.coeffs().cwiseAbs().maxCoeff();
    if (scale == 0) throw std::domain_error("Eigen quaternion inverse needs a nonzero quaternion");
    Quaterniond result = value.conjugate();
    result.coeffs() /= scale;
    result.coeffs() /= result.squaredNorm();
    result.coeffs() /= scale;
    finiteQuaternion(result);
    return result;
}
void unitQuaternion(const Quaterniond& value) {
    finiteQuaternion(value);
    if (std::abs(quaternionNorm(value) - 1.0) > 1e-9)
        throw std::domain_error("Eigen rotation needs a unit quaternion; use Eigen.Quaterniond.normalized(q)");
}
Quaterniond quaternionProduct(const Quaterniond& a, const Quaterniond& b) {
    finiteQuaternion(a);
    finiteQuaternion(b);
    Quaterniond result = a * b;
    finiteQuaternion(result);
    return result;
}
Vector3d rotate(const Quaterniond& value, const Vector3d& vector) {
    unitQuaternion(value);
    return value * vector;
}
Matrix3d toRotationMatrix(const Quaterniond& value) {
    unitQuaternion(value);
    return value.toRotationMatrix();
}
Quaterniond fromRotationMatrix(const Matrix3d& value) {
    if (!value.allFinite() || !(value.transpose() * value).isApprox(Matrix3d::Identity(), 1e-9)
        || std::abs(value.determinant() - 1.0) > 1e-9)
        throw std::domain_error("Eigen quaternion conversion needs an orthonormal rotation matrix with determinant +1");
    return quaternionNormalized(Quaterniond(value));
}
double& coefficient(Quaterniond& value, int index) { return value.coeffs()[index]; }
double coefficientCopy(const Quaterniond& value, int index) { return value.coeffs()[index]; }

class QuaternionTypeInfo : public EigenTypeInfo<Quaterniond> {
public:
    QuaternionTypeInfo() : EigenTypeInfo<Quaterniond>("/Eigen/Quaterniond") {}
    using EigenTypeInfo<Quaterniond>::getMember;
    std::vector<std::string> getMemberNames() const override { return {"w", "x", "y", "z"}; }
    Source getMember(Source item, Source id) const override {
        auto name = DataSource<std::string>::narrow(id.get());
        if (!name) return nullptr;
        const std::string member = name->get();
        const int index = member == "w" ? 3 : member == "x" ? 0 : member == "y" ? 1 : member == "z" ? 2 : -1;
        if (index < 0) return nullptr;
        Source indexSource = new ConstantDataSource<int>(index);
        if (item->isAssignable())
            return newFunctorDataSource(&coefficient, GenerateDataSource()(item.get(), indexSource.get()));
        return newFunctorDataSource(&coefficientCopy, GenerateDataSource()(item.get(), indexSource.get()));
    }
    bool decomposeTypeImpl(const Quaterniond& value, PropertyBag& bag) const override {
        if (!bag.empty()) return false;
        bag.setType(this->getTypeName());
        bag.ownProperty(new Property<double>("w", "Scalar coefficient", value.w()));
        bag.ownProperty(new Property<double>("x", "X coefficient", value.x()));
        bag.ownProperty(new Property<double>("y", "Y coefficient", value.y()));
        bag.ownProperty(new Property<double>("z", "Z coefficient", value.z()));
        return true;
    }
    bool composeTypeImpl(const PropertyBag& bag, Quaterniond& result) const override {
        if (bag.getType() != this->getTypeName() || bag.size() != 4) return false;
        Property<double> w = bag.getProperty("w"), x = bag.getProperty("x"),
                         y = bag.getProperty("y"), z = bag.getProperty("z");
        if (!w.ready() || !x.ready() || !y.ready() || !z.ready()) return false;
        // Persist coefficients exactly, including non-unit quaternions.
        result = Quaterniond(w.get(), x.get(), y.get(), z.get());
        return true;
    }
};
} // namespace
void loadQuaternionType() { RTT::types::Types()->addType(new QuaternionTypeInfo); }
void loadQuaternionConstructors() {
    type<Quaterniond>()->addConstructor(newConstructor(&quaternion));
    type<Quaterniond>()->addConstructor(new ExactConstructor<std::vector<double>>(newConstructor(&quaternionArray)));
    type<Quaterniond>()->addConstructor(new ExactConstructor<Matrix3d>(newConstructor(&fromRotationMatrix)));
}
void loadQuaternionOperators() {
    auto operators = RTT::types::OperatorRepository::Instance();
    operators->add(binary("*", &quaternionProduct));
    operators->add(binary("*", &rotate));
}
void loadQuaternionGlobals() {
    auto api = service<Quaterniond>();
    api->addOperation("identity", &zero<Quaterniond>, RTT::ClientThread);
    api->addOperation("norm", &quaternionNorm, RTT::ClientThread);
    api->addOperation("normalized", &quaternionNormalized, RTT::ClientThread);
    api->addOperation("conjugate", &quaternionConjugate, RTT::ClientThread);
    api->addOperation("inverse", &quaternionInverse, RTT::ClientThread);
    api->addOperation("rotate", &rotate, RTT::ClientThread);
    api->addOperation("toRotationMatrix", &toRotationMatrix, RTT::ClientThread);
    api->addOperation("fromRotationMatrix", &fromRotationMatrix, RTT::ClientThread);
}
} // namespace Eigen::rtt
