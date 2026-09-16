// Copyright  (C)  2008  Ruben Smits <ruben dot smits at mech dot kuleuven dot be>

// Author: Ruben Smits <ruben dot smits at mech dot kuleuven dot be>
// Maintainer: Ruben Smits <ruben dot smits at mech dot kuleuven dot be>

// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

#include <eigen_typekit/eigen_typekit.hpp>
#include <rtt/Property.hpp>
#include <rtt/PropertyBag.hpp>
#include <rtt/types/TemplateTypeInfo.hpp>
#include <rtt/types/Operators.hpp>
#include <rtt/types/OperatorTypes.hpp>
#include <rtt/types/TemplateConstructor.hpp>
#include <rtt/types/Types.hpp>
#include <rtt/Logger.hpp>
#include <rtt/internal/DataSources.hpp>
#include <rtt/internal/mystd.hpp>
#include <rtt/os/StartStopManager.hpp>
#include <rtt/types/TypekitRepository.hpp>
#include <rtt/internal/FusedFunctorDataSource.hpp>
#include <rtt/internal/DataSourceGenerator.hpp>

// Follow OroGen on MSVC: instantiate templates in consumers. Explicit port
// exports fail with C2908 for their covariant getEndpoint() return types.
#ifndef _MSC_VER
#define DECLARE_RTT_VECTOR_EXPORTS( VectorType ) \
template class RTT_EXPORT RTT::internal::DataSourceTypeInfo< VectorType >; \
template class RTT_EXPORT RTT::internal::DataSource< VectorType >; \
template class RTT_EXPORT RTT::internal::AssignableDataSource< VectorType >; \
template class RTT_EXPORT RTT::internal::AssignCommand< VectorType >; \
template class RTT_EXPORT RTT::internal::ValueDataSource< VectorType >; \
template class RTT_EXPORT RTT::internal::ConstantDataSource< VectorType >; \
template class RTT_EXPORT RTT::internal::ReferenceDataSource< VectorType >; \
template class RTT_EXPORT RTT::OutputPort< VectorType >; \
template class RTT_EXPORT RTT::InputPort< VectorType >; \
template class RTT_EXPORT RTT::Property< VectorType >; \
template class RTT_EXPORT RTT::Attribute< VectorType >; \
template class RTT_EXPORT RTT::Constant< VectorType >;

DECLARE_RTT_VECTOR_EXPORTS( Eigen::VectorXd )
DECLARE_RTT_VECTOR_EXPORTS( Eigen::Vector2d )
DECLARE_RTT_VECTOR_EXPORTS( Eigen::Vector3d )
DECLARE_RTT_VECTOR_EXPORTS( Eigen::Vector4d )
DECLARE_RTT_VECTOR_EXPORTS( Eigen::Vector6d )
DECLARE_RTT_VECTOR_EXPORTS( Eigen::Quaterniond )

#define DECLARE_RTT_MATRIX_EXPORTS( MatrixType ) \
template class RTT_EXPORT RTT::internal::DataSourceTypeInfo< MatrixType >; \
template class RTT_EXPORT RTT::internal::DataSource< MatrixType >; \
template class RTT_EXPORT RTT::internal::AssignableDataSource< MatrixType >; \
template class RTT_EXPORT RTT::internal::AssignCommand< MatrixType >; \
template class RTT_EXPORT RTT::internal::ValueDataSource< MatrixType >; \
template class RTT_EXPORT RTT::internal::ConstantDataSource< MatrixType >; \
template class RTT_EXPORT RTT::internal::ReferenceDataSource< MatrixType >; \
template class RTT_EXPORT RTT::OutputPort< MatrixType >; \
template class RTT_EXPORT RTT::InputPort< MatrixType >; \
template class RTT_EXPORT RTT::Property< MatrixType >; \
template class RTT_EXPORT RTT::Attribute< MatrixType >; \
template class RTT_EXPORT RTT::Constant< MatrixType >;

DECLARE_RTT_MATRIX_EXPORTS( Eigen::MatrixXd )
DECLARE_RTT_MATRIX_EXPORTS( Eigen::Matrix2d )
DECLARE_RTT_MATRIX_EXPORTS( Eigen::Matrix3d )
DECLARE_RTT_MATRIX_EXPORTS( Eigen::Matrix4d )

#endif

#include "eigen_scripting.hpp"

namespace Eigen {
std::string EigenTypekitPlugin::getName() { return "eigen_typekit"; }
bool EigenTypekitPlugin::loadTypes() {
    rtt::loadVectorTypes();
    rtt::loadMatrixTypes();
    rtt::loadQuaternionType();
    return true;
}
bool EigenTypekitPlugin::loadConstructors() {
    rtt::loadVectorConstructors();
    rtt::loadMatrixConstructors();
    rtt::loadQuaternionConstructors();
    return true;
}
bool EigenTypekitPlugin::loadOperators() {
    rtt::loadVectorOperators();
    rtt::loadMatrixOperators();
    rtt::loadQuaternionOperators();
    return true;
}
bool EigenTypekitPlugin::loadGlobals() {
    rtt::loadVectorGlobals();
    rtt::loadMatrixGlobals();
    rtt::loadQuaternionGlobals();
    return true;
}
}
ORO_TYPEKIT_PLUGIN(Eigen::EigenTypekitPlugin)
