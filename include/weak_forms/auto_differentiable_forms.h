// ---------------------------------------------------------------------
//
// Copyright (C) 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#ifndef dealii_weakforms_auto_differentiable_forms_h
#define dealii_weakforms_auto_differentiable_forms_h


// ======================================================================
// TODO: Remove this header. Its superceded by the self-linearizing forms
// ======================================================================


// #include <deal.II/base/config.h>
#include <weak_forms/config.h>

// #include <deal.II/differentiation/ad.h>

// #include <weak_forms/assembler.h>
// #include <weak_forms/bilinear_forms.h>
// #include <weak_forms/symbolic_integral.h>
// #include <weak_forms/linear_forms.h>
// #include <weak_forms/self_linearizing_forms.h>
// // #include <weak_forms/spaces.h>

// #include <functional>
// #include <tuple>


// WEAK_FORMS_NAMESPACE_OPEN


// namespace WeakForms
// {
//   namespace AutoDifferentiation
//   {
//     /**
//      * OP: (AutoDifferentiableFunctor)
//      *
//      * First derivatives of this form produce a ResidualForm.
//      */
//     template <int                                   dim,
//               enum Differentiation::AD::NumberTypes ADNumberTypeCode,
//               typename ScalarType = double>
//     class EnergyFunctional
//     {
//       using ADHelper_t =
//         Differentiation::AD::ScalarFunction<dim, ADNumberTypeCode,
//         ScalarType>;

//       template <typename... SymbolicOpsSubSpaceFieldSolution>
//       using SelfLinearizationHelper_t =
//         WeakForms::SelfLinearization::internal::SelfLinearizationHelper<
//           SymbolicOpsSubSpaceFieldSolution...>;

//       template <typename NumberType, typename...
//       SymbolicOpsSubSpaceFieldSolution> using functor_arguments_t =
//         typename
//         SelfLinearizationHelper_t<SymbolicOpsSubSpaceFieldSolution...>::
//           template type_list_functor_arguments<NumberType>;

//     public:
//       using ad_type = typename ADHelper_t::ad_type;

//       // template<typename... SymbolicOpsSubSpaceFieldSolution>
//       // using ad_functor_type =
//       // std::function<ad_type(functor_arguments_t<ad_type,
//       // SymbolicOpsSubSpaceFieldSolution...>)>;
//       template <typename... SymbolicOpsSubSpaceFieldSolution>
//       using ad_functor_type =
//         std::function<ad_type(typename SymbolicOpsSubSpaceFieldSolution::
//                                 template value_type<ad_type>...)>;

//       // TODO: Try to introduce ADNumberTypeCode etc. here.
//       template <typename... FieldArgs>
//       EnergyFunctional(const ad_functor_type<FieldArgs...> &functor,
//                        const FieldArgs &... symbolic_op_field_args)
//       {
//         using Functor                   = ad_functor_type<FieldArgs...>;
//         using FieldSolutionOpCollection = std::tuple<FieldArgs...>;

//         // const Functor functor (functor);
//         const FieldSolutionOpCollection
//         field_args(symbolic_op_field_args...);

//         // TODO:
//         // - Initialise AD helper
//         // - Update solution
//         // - Linear form contributions
//         // - Bilinear form contributions

//         // const unsigned int n_independent_variables =
//         //   SymmetricTensor<2, dim>::n_independent_components +
//         //   Tensor<1, dim>::n_independent_components;
//         // ScalarFunction<dim, ...> ad_helper(n_independent_variables);
//         // using ADNumberType = typename ADHelper::ad_type;

//         // ad_helper.register_independent_variable(H, H_dofs);
//         // ad_helper.register_independent_variable(C, C_dofs);

//         // const SymmetricTensor<2, dim, ADNumberType> C_AD =
//         //   ad_helper.get_sensitive_variables(C_dofs);
//         // const Tensor<1, dim, ADNumberType> H_AD =
//         //   ad_helper.get_sensitive_variables(H_dofs);


//         // const ADNumberType psi = 0.5 * mu_e *
//         //                            (1.0 + std::tanh((H_AD * H_AD) /
//         100.0)) *
//         //                            (trace(C_AD) - dim - 2 * std::log(J)) +
//         //                          lambda_e * std::log(J) * std::log(J) -
//         //                          0.5 * mu_0 * mu_r * J * H_AD * C_inv_AD *
//         //                          H_AD;
//         // // Register the definition of the total stored energy
//         // ad_helper.register_dependent_variable(psi);

//         // Vector<double>     Dpsi(ad_helper.n_dependent_variables());
//         // FullMatrix<double> D2psi(ad_helper.n_dependent_variables(),
//         //                          ad_helper.n_independent_variables());
//         // const double       psi = ad_helper.compute_value();
//         // ad_helper.compute_gradient(Dpsi);
//         // ad_helper.compute_hessian(D2psi);


//         // const SymmetricTensor<2, dim> S =
//         //   2.0 * ad_helper.extract_gradient_component(Dpsi, C_dofs);
//         // const SymmetricTensor<4, dim> HH =
//         //   4.0 * ad_helper.extract_hessian_component(D2psi, C_dofs,
//         C_dofs);
//       }

//       // ===== Section: Integration =====

//       auto
//       dV() const
//       {
//         return integrate(*this, VolumeIntegral());
//       }

//       auto
//       dV(const typename VolumeIntegral::subdomain_t subdomain) const
//       {
//         return dV(std::set<typename VolumeIntegral::subdomain_t>{subdomain});
//       }

//       auto
//       dV(const std::set<typename VolumeIntegral::subdomain_t> &subdomains)
//       const
//       {
//         return integrate(*this, VolumeIntegral(subdomains));
//       }

//       auto
//       dA() const
//       {
//         return integrate(*this, BoundaryIntegral());
//       }

//       auto
//       dA(const typename BoundaryIntegral::subdomain_t boundary) const
//       {
//         return dA(std::set<typename
//         BoundaryIntegral::subdomain_t>{boundary});
//       }

//       auto
//       dA(const std::set<typename BoundaryIntegral::subdomain_t> &boundaries)
//         const
//       {
//         return integrate(*this, BoundaryIntegral(boundaries));
//       }

//       auto
//       dI() const
//       {
//         return integrate(*this, InterfaceIntegral());
//       }

//       auto
//       dI(const typename InterfaceIntegral::subdomain_t interface) const
//       {
//         return dI(std::set<typename
//         InterfaceIntegral::subdomain_t>{interface});
//       }

//       auto
//       dI(const std::set<typename InterfaceIntegral::subdomain_t> &interfaces)
//         const
//       {
//         return integrate(*this, InterfaceIntegral(interfaces));
//       }

//     private:
//       // Provide access to accumulation function
//       template <int dim2,
//                 int spacedim,
//                 typename NumberType,
//                 bool use_vectorization>
//       friend class WeakForms::AssemblerBase;

//       template <enum internal::AccumulationSign Sign,
//                 typename AssemblerType,
//                 typename IntegralType>
//       void
//       accumulate_into(AssemblerType &     assembler,
//                       const IntegralType &integral_operation) const
//       {
//         std::cout << "HERE!" << std::endl;

//         // ADHelper_t &ad_helper = assembler.ad_sd_cache.template
//         // get_or_add_object_with_name<ADHelper_t>("tmp");

//         // explicit SymbolicOp(const IntegralType & integral_operation,
//         //            const IntegrandType &integrand);

//         throw;
//       }

//       // This object might be temporary. So this must be a shared
//       // pointer so that we can hand over ownership to the assembler.
//       // std::shared_ptr<ADHelper_t> ad_helper;


//       // std::vector<std::function<void()>> linear_form_integrand_plus_ops;
//       // std::vector<std::function<void()>> linear_form_integrand_minus_ops;
//       // std::vector<std::function<void()>> bilinear_form_integrand_plus_ops;
//       // std::vector<std::function<void()>>
//       bilinear_form_integrand_minus_ops;
//     };

//     // /**
//     //  * OP: (Variation, SymbolicFunctor)
//     //  *
//     //  * This class gets converted into a LinearForm.
//     //  * First derivatives of this form produce a BilinearForm through the
//     //  * LinearizationForm
//     //  */
//     // class ResidualForm
//     // {};

//     // /**
//     //  * OP: (Variation, SymbolicFunctor, Linearization)
//     //  *
//     //  * This class gets converted into a LinearForm.
//     //  * First derivatives of this form produce a BilinearForm through the
//     //  * LinearizationForm
//     //  */
//     // class LinearizationForm
//     // {
//     // private:
//     //   // friend EnergyFunctional;
//     //   // friend ResidualForm;
//     //   LinearizationForm() = default;
//     // };
//   } // namespace AutoDifferentiation

// } // namespace WeakForms



// /* ======================== Convenience functions ======================== */



// namespace WeakForms
// {
//   //   // template <typename TestSpaceOp, typename TrialSpaceOp>
//   //   // BilinearForm<TestSpaceOp, NoOp, TrialSpaceOp>
//   //   // bilinear_form(const TestSpaceOp & test_space_op,
//   //   //               const TrialSpaceOp &trial_space_op)
//   //   // {
//   //   //   return BilinearForm<TestSpaceOp, NoOp,
//   TrialSpaceOp>(test_space_op,
//   //   // trial_space_op);
//   //   // }

//   template <int                                   dim,
//             enum Differentiation::AD::NumberTypes ADNumberTypeCode,
//             typename ScalarType = double,
//             typename... FieldArgs>
//   AutoDifferentiation::EnergyFunctional<dim, ADNumberTypeCode, ScalarType>
//   ad_energy_functional_form(
//     const typename AutoDifferentiation::EnergyFunctional<
//       dim,
//       ADNumberTypeCode,
//       ScalarType>::template ad_functor_type<FieldArgs...> &functor_op,
//     const FieldArgs &... dependent_fields)
//   {
//     return AutoDifferentiation::
//       EnergyFunctional<dim, ADNumberTypeCode, ScalarType>(functor_op,
//                                                           dependent_fields...);
//   }

// } // namespace WeakForms



// /* ==================== Specialization of type traits ==================== */



// #ifndef DOXYGEN


// namespace WeakForms
// {
//   template <int                                   dim,
//             enum Differentiation::AD::NumberTypes ADNumberTypeCode,
//             typename ScalarType>
//   struct is_self_linearizing_form<
//     AutoDifferentiation::EnergyFunctional<dim, ADNumberTypeCode, ScalarType>>
//     : std::true_type
//   {};

// } // namespace WeakForms


// #endif // DOXYGEN


// WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_auto_differentiable_forms_h
