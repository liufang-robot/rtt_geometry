// SPDX-License-Identifier: LGPL-2.1-or-later
#pragma once

#include <rtt/types/TemplateTypeInfo.hpp>
#include <rtt/internal/DataSources.hpp>
#include <rtt/internal/DataSourceGenerator.hpp>
#include <rtt/internal/FusedFunctorDataSource.hpp>
// The public typekit header declares extern templates for RTT classes that are
// already included. Suppress implicit (possibly hidden) copies in these helper
// translation units; eigen_typekit.cpp owns their exported instantiations.
#include <eigen_typekit/eigen_typekit.hpp>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace Eigen::rtt {
using RTT::base::DataSourceBase;
using RTT::internal::DataSource;
using RTT::internal::AssignableDataSource;
using RTT::internal::ConstantDataSource;
using RTT::internal::ValueDataSource;
using RTT::internal::GenerateDataSource;
using RTT::internal::newFunctorDataSource;
using RTT::Property;
using RTT::PropertyBag;
using RTT::types::TypeInfo;
using Source = DataSourceBase::shared_ptr;

template<class T> T initialValue() {
    if constexpr (std::is_same_v<T, Quaterniond>) return Quaterniond::Identity();
    else if constexpr (T::SizeAtCompileTime == Dynamic) return T();
    else return T::Zero();
}

inline void checkShape(int rows, int cols) {
    if (rows < 0 || cols < 0 || (cols && rows > std::numeric_limits<int>::max() / cols))
        throw std::invalid_argument("Eigen dimensions must be nonnegative and their product fit Int32");
}

template<class T> void checkShape(int rows, int cols) {
    checkShape(rows, cols);
    if constexpr (T::RowsAtCompileTime != Dynamic)
        if (rows != T::RowsAtCompileTime)
            throw std::invalid_argument("Eigen row count does not match the fixed type");
    if constexpr (T::ColsAtCompileTime != Dynamic)
        if (cols != T::ColsAtCompileTime)
            throw std::invalid_argument("Eigen column count does not match the fixed type");
}

inline void checkIndex(int index, Index size) {
    if (index < 0 || index >= size)
        throw std::out_of_range("Eigen index " + std::to_string(index) +
                                " is outside [0, " + std::to_string(size) + ")");
}

template<class T> int sizeOf(const T& value) { return static_cast<int>(value.size()); }
template<class T> int rowsOf(const T& value) { return static_cast<int>(value.rows()); }
template<class T> int colsOf(const T& value) { return static_cast<int>(value.cols()); }
template<class T> std::vector<double> matrixData(const T& value) {
    std::vector<double> result;
    result.reserve(value.size());
    for (Index row = 0; row < value.rows(); ++row)
        for (Index col = 0; col < value.cols(); ++col) result.push_back(value(row, col));
    return result;
}
template<class T> double& vectorItem(T& value, int index) {
    checkIndex(index, value.size());
    return value[index];
}
template<class T> double vectorItemCopy(const T& value, int index) {
    checkIndex(index, value.size());
    return value[index];
}
template<class T> double& matrixItem(T& value, int row, int col) {
    checkIndex(row, value.rows());
    checkIndex(col, value.cols());
    return value(row, col);
}
template<class T> double matrixItemCopy(const T& value, int row, int col) {
    checkIndex(row, value.rows());
    checkIndex(col, value.cols());
    return value(row, col);
}

// A row reads as VectorXd, but its element factory retains the matrix source.
// In particular, m[r][c] must never assign into a temporary row copy.
struct MatrixRowSource {
    virtual ~MatrixRowSource() = default;
    virtual Source element(DataSource<int>::shared_ptr column) const = 0;
};

template<class T> class MatrixRowDataSource : public DataSource<VectorXd>, public MatrixRowSource {
    typename DataSource<T>::shared_ptr matrix;
    DataSource<int>::shared_ptr row;
    mutable VectorXd cached;
public:
    MatrixRowDataSource(typename DataSource<T>::shared_ptr source, DataSource<int>::shared_ptr index)
        : matrix(source), row(index) {}
    VectorXd get() const override {
        matrix->evaluate();
        const int index = row->get();
        checkIndex(index, matrix->rvalue().rows());
        return cached = matrix->rvalue().row(index).transpose();
    }
    VectorXd value() const override { return cached; }
    const VectorXd& rvalue() const override { return cached; }
    void reset() override { matrix->reset(); row->reset(); }
    MatrixRowDataSource* clone() const override { return new MatrixRowDataSource(matrix, row); }
    DataSource<VectorXd>* copy(std::map<const DataSourceBase*, DataSourceBase*>& replacements) const override {
        auto found = replacements.find(this);
        if (found != replacements.end()) return DataSource<VectorXd>::narrow(found->second);
        auto result = new MatrixRowDataSource(matrix->copy(replacements), row->copy(replacements));
        replacements[this] = result;
        return result;
    }
    Source element(DataSource<int>::shared_ptr column) const override {
        if (matrix->isAssignable())
            return newFunctorDataSource(&matrixItem<T>, GenerateDataSource()(matrix.get(), row.get(), column.get()));
        return newFunctorDataSource(&matrixItemCopy<T>, GenerateDataSource()(matrix.get(), row.get(), column.get()));
    }
};

// RTT-created values have deterministic defaults. This does not change Eigen's
// default constructor or initialization policy in downstream C++ code.
template<class T> class EigenTypeInfo : public RTT::types::TemplateTypeInfo<T, false>,
                                      public RTT::types::MemberFactory {
    using Base = RTT::types::TemplateTypeInfo<T, false>;
public:
    explicit EigenTypeInfo(const std::string& name) : Base(name) {}
    bool installTypeInfoObject(TypeInfo* type) override {
        auto self = boost::dynamic_pointer_cast<EigenTypeInfo>(this->getSharedPtr());
        Base::installTypeInfoObject(type);
        type->setMemberFactory(self);
        type->setStreamFactory(self);
        return false;
    }
    RTT::base::AttributeBase* buildVariable(std::string name) const override {
        return new RTT::Attribute<T>(name,
            new RTT::internal::UnboundDataSource<ValueDataSource<T>>(initialValue<T>()));
    }
    Source buildValue() const override { return new ValueDataSource<T>(initialValue<T>()); }
    RTT::base::AttributeBase* buildAttribute(std::string name, Source source) const override {
        if (!source) source = buildValue();
        return Base::buildAttribute(name, source);
    }
    RTT::base::PropertyBase* buildProperty(const std::string& name, const std::string& desc,
                                         Source source = nullptr) const override {
        if (!source) source = buildValue();
        if (!AssignableDataSource<T>::narrow(source.get())) return nullptr;
        return Base::buildProperty(name, desc, source);
    }
    std::ostream& write(std::ostream& stream, Source source) const override {
        auto typed = DataSource<T>::narrow(source.get());
        if (typed) {
            const auto& value = typed->rvalue();
            if constexpr (std::is_same_v<T, Quaterniond>)
                stream << "(w=" << value.w() << ", x=" << value.x()
                       << ", y=" << value.y() << ", z=" << value.z() << ')';
            else stream << value;
        }
        return stream;
    }
    std::istream& read(std::istream& stream, Source) const override {
        // Text display is not a serialization format. Use constructors or CPF.
        stream.setstate(std::ios::failbit);
        return stream;
    }
    bool isStreamable() const override { return true; }
    Source getMember(Source item, const std::string& name) const override {
        // Keep numeric property names working for C++ reflection clients.
        if (!name.empty() && name.find_first_not_of("0123456789") == std::string::npos) {
            try { return getMember(item, new ConstantDataSource<int>(std::stoi(name))); }
            catch (const std::out_of_range&) { return nullptr; }
        }
        return getMember(item, new ConstantDataSource<std::string>(name));
    }
    virtual Source getMember(Source item, Source id) const override = 0;
};

template<class T> class VectorTypeInfo : public EigenTypeInfo<T> {
public:
    explicit VectorTypeInfo(const std::string& name) : EigenTypeInfo<T>(name) {}
    using EigenTypeInfo<T>::getMember;
    std::vector<std::string> getMemberNames() const override { return {"size", "capacity"}; }
    Source getMember(Source item, Source id) const override {
        if (auto name = DataSource<std::string>::narrow(id.get())) {
            if (name->get() == "size" || name->get() == "capacity")
                return newFunctorDataSource(&sizeOf<T>, GenerateDataSource()(item.get()));
        }
        DataSource<int>::shared_ptr index = DataSource<int>::narrow(id.get());
        if (!index) return nullptr;
        if (auto row = dynamic_cast<MatrixRowSource*>(item.get())) return row->element(index);
        if (item->isAssignable())
            return newFunctorDataSource(&vectorItem<T>, GenerateDataSource()(item.get(), index.get()));
        return newFunctorDataSource(&vectorItemCopy<T>, GenerateDataSource()(item.get(), index.get()));
    }
    bool resize(Source source, int size) const override {
        auto writable = AssignableDataSource<T>::narrow(source.get());
        if (!writable) return false;
        checkShape<T>(size, 1);
        if constexpr (T::SizeAtCompileTime == Dynamic)
            writable->set().conservativeResizeLike(T::Zero(size));
        writable->updated();
        return true;
    }
    bool decomposeTypeImpl(const T& value, PropertyBag& bag) const override {
        if (!bag.empty()) return false;
        bag.setType(this->getTypeName());
        for (Index i = 0; i < value.size(); ++i)
            bag.ownProperty(new Property<double>(std::to_string(i + 1), "Vector element", value[i]));
        return true;
    }
    bool composeTypeImpl(const PropertyBag& bag, T& result) const override {
        if (bag.getType() != this->getTypeName()) return false;
        const int size = static_cast<int>(bag.size());
        try { checkShape<T>(size, 1); } catch (const std::invalid_argument&) { return false; }
        T value = T::Zero(size);
        for (int i = 0; i < size; ++i) {
            Property<double> element = bag.getProperty(std::to_string(i + 1));
            if (!element.ready()) return false;
            value[i] = element.get();
        }
        result = value;
        return true;
    }
};

template<class T> class MatrixTypeInfo : public EigenTypeInfo<T> {
public:
    explicit MatrixTypeInfo(const std::string& name) : EigenTypeInfo<T>(name) {}
    using EigenTypeInfo<T>::getMember;
    std::vector<std::string> getMemberNames() const override { return {"rows", "cols", "size", "data"}; }
    Source getMember(Source item, Source id) const override {
        if (auto name = DataSource<std::string>::narrow(id.get())) {
            if (name->get() == "rows") return newFunctorDataSource(&rowsOf<T>, GenerateDataSource()(item.get()));
            if (name->get() == "cols") return newFunctorDataSource(&colsOf<T>, GenerateDataSource()(item.get()));
            if (name->get() == "size") return newFunctorDataSource(&sizeOf<T>, GenerateDataSource()(item.get()));
            if (name->get() == "data") return newFunctorDataSource(&matrixData<T>, GenerateDataSource()(item.get()));
        }
        DataSource<int>::shared_ptr index = DataSource<int>::narrow(id.get());
        if (!index) return nullptr;
        return new MatrixRowDataSource<T>(DataSource<T>::narrow(item.get()), index);
    }
    bool decomposeTypeImpl(const T& value, PropertyBag& bag) const override {
        if (!bag.empty()) return false;
        bag.setType(this->getTypeName());
        // Preserve the old numbered-row format. A zero-row matrix needs its
        // column count recorded explicitly because there are no rows to infer it.
        if (value.rows() == 0) bag.ownProperty(new Property<int>("cols", "Empty matrix columns", value.cols()));
        for (Index i = 0; i < value.rows(); ++i) {
            Property<VectorXd> row(std::to_string(i + 1), "Matrix row", value.row(i).transpose());
            Source decomposed = row.getTypeInfo()->decomposeType(row.getDataSource());
            auto rowBag = AssignableDataSource<PropertyBag>::narrow(decomposed.get());
            if (!rowBag) return false;
            bag.ownProperty(new Property<PropertyBag>(row.getName(), row.getDescription(), rowBag));
        }
        return true;
    }
    bool composeTypeImpl(const PropertyBag& bag, T& result) const override {
        if (bag.getType() != this->getTypeName()) return false;
        int rows = static_cast<int>(bag.size()), cols = 0;
        Property<int> emptyColumns = bag.getProperty("cols");
        if (emptyColumns.ready()) {
            if (rows != 1) return false;
            rows = 0;
            cols = emptyColumns.get();
        }
        std::vector<VectorXd> values;
        for (int i = 0; i < rows; ++i) {
            Property<PropertyBag> rowBag = bag.getProperty(std::to_string(i + 1));
            Property<VectorXd> row("row", "Matrix row");
            if (!rowBag.ready() || !row.compose(rowBag) || !row.ready()) return false;
            if (i == 0) cols = static_cast<int>(row.get().size());
            if (row.get().size() != cols) return false;
            values.push_back(row.get());
        }
        try { checkShape<T>(rows, cols); } catch (const std::invalid_argument&) { return false; }
        T value = T::Zero(rows, cols);
        for (int i = 0; i < rows; ++i) value.row(i) = values[i].transpose();
        result = value;
        return true;
    }
};
} // namespace Eigen::rtt
