// ---------------------------------------------------------------------
//
// Copyright (C) 2021 - 2022 by Jean-Paul Pelteret
//
// This file is part of the Weak forms for deal.II library.
//
// The Weak forms for deal.II library is free software; you can use it,
// redistribute it, and/or modify it under the terms of the GNU Lesser
// General Public License as published by the Free Software Foundation;
// either version 3.0 of the License, or (at your option) any later
// version. The full text of the license can be found in the file LICENSE
// at the top level of the Weak forms for deal.II distribution.
//
// ---------------------------------------------------------------------

#ifndef dealii_weakforms_ad_sd_functor_internal_h
#define dealii_weakforms_ad_sd_functor_internal_h

#include <deal.II/base/config.h>

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/differentiation/ad.h>
#include <deal.II/differentiation/sd.h>

#include <weak_forms/config.h>
#include <weak_forms/differentiation.h>
#include <weak_forms/solution_extraction_data.h>
#include <weak_forms/spaces.h>
#include <weak_forms/types.h>
#include <weak_forms/utilities.h>

#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Operators
  {
    namespace internal
    {
      template <typename SymbolicSpaceOp,
                typename ScalarType = double,
                typename            = typename std::enable_if<
                  is_or_has_test_function_op<SymbolicSpaceOp>::value ||
                  is_field_solution<SymbolicSpaceOp>::value>>
      struct SpaceOpComponentInfo
      {
        using value_type =
          typename SymbolicSpaceOp::template value_type<ScalarType>;
        static constexpr unsigned int n_components =
          WeakForms::Utilities::ValueHelper<value_type>::n_components;
        using extractor_type = typename WeakForms::Utilities::ValueHelper<
          value_type>::extractor_type;
      };



      // Ensure that template arguments contain no duplicates.
      // Adapted from https://stackoverflow.com/a/34122593
      namespace TemplateRestrictions
      {
        template <typename T, typename... List>
        struct IsContained;


        template <typename T, typename Head, typename... Tail>
        struct IsContained<T, Head, Tail...>
        {
          static constexpr bool value =
            std::is_same<T, Head>::value || IsContained<T, Tail...>::value;
        };


        template <typename T>
        struct IsContained<T>
        {
          static constexpr bool value = false;
        };


        template <typename... List>
        struct IsUnique;


        template <typename Head, typename... Tail>
        struct IsUnique<Head, Tail...>
        {
          static constexpr bool value =
            !IsContained<Head, Tail...>::value && IsUnique<Tail...>::value;
        };


        template <>
        struct IsUnique<>
        {
          static constexpr bool value = true;
        };


        template <typename... Ts>
        struct EnforceNoDuplicates
        {
          static_assert(IsUnique<Ts...>::value, "No duplicate types allowed.");

          static constexpr bool value = IsUnique<Ts...>::value;
        };


        // Check that all types in a parameter pack are not tuples
        // without using C++17 fold expressions...
        // https://stackoverflow.com/a/29671981
        // https://stackoverflow.com/a/29603896
        // https://stackoverflow.com/a/32234520

        template <typename>
        struct is_tuple : std::false_type
        {};

        template <typename... T>
        struct is_tuple<std::tuple<T...>> : std::true_type
        {};

        template <bool...>
        struct bool_pack;
        template <bool... bs>
        using all_true =
          std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;
        template <bool... bs>
        using all_false =
          std::is_same<bool_pack<bs..., false>, bool_pack<false, bs...>>;
        template <typename... Ts>
        using are_tuples = all_true<is_tuple<Ts>::value...>;
        template <typename... Ts>
        using are_not_tuples = all_false<is_tuple<Ts>::value...>;

        template <typename T, typename... Us>
        struct is_subspace_field_solution_op
        {
          static constexpr bool value =
            is_subspace_field_solution_op<T>::value &&
            is_subspace_field_solution_op<Us...>::value;
        };

        // Scalar and Vector subspaces
        template <template <class> class SubSpaceViewsType,
                  typename SpaceType,
                  enum WeakForms::Operators::SymbolicOpCodes OpCode,
                  types::solution_index                      solution_index>
        struct is_subspace_field_solution_op<WeakForms::Operators::SymbolicOp<
          SubSpaceViewsType<SpaceType>,
          OpCode,
          void,
          WeakForms::internal::SolutionIndex<solution_index>>>
        {
          static constexpr bool value =
            is_field_solution<SubSpaceViewsType<SpaceType>>::value &&
            is_subspace_view<SubSpaceViewsType<SpaceType>>::value;
        };

        // Tensor and SymmetricTensor subspaces
        template <template <int, class> class SubSpaceViewsType,
                  int rank,
                  typename SpaceType,
                  enum WeakForms::Operators::SymbolicOpCodes OpCode,
                  types::solution_index                      solution_index>
        struct is_subspace_field_solution_op<WeakForms::Operators::SymbolicOp<
          SubSpaceViewsType<rank, SpaceType>,
          OpCode,
          void,
          WeakForms::internal::SolutionIndex<solution_index>>>
        {
          static constexpr bool value =
            is_field_solution<SubSpaceViewsType<rank, SpaceType>>::value &&
            is_subspace_view<SubSpaceViewsType<rank, SpaceType>>::value;
        };

        template <typename T>
        struct is_subspace_field_solution_op<T> : std::false_type
        {};

        template <typename... FieldArgs>
        struct EnforceIsSymbolicOpSubspaceFieldSolution
        {
          static_assert(
            is_subspace_field_solution_op<FieldArgs...>::value,
            "Template arguments must be unary operation subspace field solutions. "
            "You might have used a test function or trial solution, or perhaps "
            "have not used a sub-space extractor.");

          static constexpr bool value = true;
        };
      } // namespace TemplateRestrictions

    } // namespace internal
  }   // namespace Operators


  // =========================
  // === SD IMPLEMENTATION ===
  // =========================

#ifdef DEAL_II_WITH_SYMENGINE

  namespace Operators
  {
    namespace internal
    {
      // ===================
      // SD helper functions
      // ===================

      inline std::string
      replace_protected_characters(const std::string &name)
      {
        // Allow SymEngine to parse this field as a string:
        // Required for deserialization.
        // It gets confused when there are numbers in the string name, and
        // we have numbers and some protected characters in the expression
        // name.
        std::string out = name;
        const auto  replace_chars =
          [&out](const char &old_char, const char &new_char)
        { std::replace(out.begin(), out.end(), old_char, new_char); };
        // replace_chars('0', 'A');
        // replace_chars('1', 'B');
        // replace_chars('2', 'C');
        // replace_chars('3', 'D');
        // replace_chars('4', 'E');
        // replace_chars('5', 'F');
        // replace_chars('6', 'G');
        // replace_chars('7', 'H');
        // replace_chars('8', 'I');
        // replace_chars('9', 'J');
        replace_chars(' ', '_');
        replace_chars('(', '_');
        replace_chars(')', '_');
        replace_chars('{', '_');
        replace_chars('}', '_');

        return out;
      }

      template <typename ReturnType>
      typename std::enable_if<
        std::is_same<ReturnType, Differentiation::SD::Expression>::value,
        ReturnType>::type
      make_symbolic(const std::string &name)
      {
        return Differentiation::SD::make_symbol(name);
      }

      template <typename ReturnType>
      typename std::enable_if<
        std::is_same<ReturnType,
                     Tensor<ReturnType::rank,
                            ReturnType::dimension,
                            Differentiation::SD::Expression>>::value,
        ReturnType>::type
      make_symbolic(const std::string &name)
      {
        constexpr int rank = ReturnType::rank;
        constexpr int dim  = ReturnType::dimension;
        return Differentiation::SD::make_tensor_of_symbols<rank, dim>(name);
      }

      template <typename ReturnType>
      typename std::enable_if<
        std::is_same<ReturnType,
                     SymmetricTensor<ReturnType::rank,
                                     ReturnType::dimension,
                                     Differentiation::SD::Expression>>::value,
        ReturnType>::type
      make_symbolic(const std::string &name)
      {
        constexpr int rank = ReturnType::rank;
        constexpr int dim  = ReturnType::dimension;
        return Differentiation::SD::make_symmetric_tensor_of_symbols<rank, dim>(
          name);
      }

      template <typename ExpressionType, typename SymbolicOpField>
      typename SymbolicOpField::template value_type<ExpressionType>
      make_symbolic(const SymbolicOpField &    field,
                    const SymbolicDecorations &decorator)
      {
        using ReturnType =
          typename SymbolicOpField::template value_type<ExpressionType>;

        const std::string name = Utilities::get_deal_II_prefix() + "Field_" +
                                 field.as_ascii(decorator);
        // return make_symbolic<ReturnType>(name);
        return make_symbolic<ReturnType>(replace_protected_characters(name));
      }

    } // namespace internal
  }   // namespace Operators


  inline void
  assertOptimizerSettings(
    const enum Differentiation::SD::OptimizerType     optimization_method,
    const enum Differentiation::SD::OptimizationFlags optimization_flags)
  {
    if (optimization_method != Differentiation::SD::OptimizerType::llvm)
      return;

    // Adding this flag doesn't return any benefit (there's actualy only some
    // extra overhead in SymEngine) so let's not allow it.
    const bool use_cse_opt =
      static_cast<int>(optimization_flags &
                       Differentiation::SD::OptimizationFlags::optimize_cse);
    Assert(
      use_cse_opt == false,
      ExcMessage(
        "The optimization setting should not include OptimizationFlags::optimize_cse when the LLVM optimizer is used."));
  }

#endif // DEAL_II_WITH_SYMENGINE


  // ================================
  // === AD and SD IMPLEMENTATION ===
  // ================================

  namespace Operators
  {
    namespace internal
    {
      template <typename... SymbolicOpsSubSpaceFieldSolution>
      struct SymbolicOpsSubSpaceFieldSolutionHelper
      {
        // ===================
        // AD type definitions
        // ===================

        using field_args_t = std::tuple<SymbolicOpsSubSpaceFieldSolution...>;
        using field_extractors_t =
          std::tuple<typename internal::SpaceOpComponentInfo<
            SymbolicOpsSubSpaceFieldSolution>::extractor_type...>;


        // ===================
        // SD type definitions
        // ===================
        template <typename ScalarType>
        using field_values_t =
          std::tuple<typename SymbolicOpsSubSpaceFieldSolution::
                       template value_type<ScalarType>...>;

        // Typical use case expects FunctionType to be an SD:Expression,
        // or a tensor of SD:Expressions. ScalarType should be a scalar
        // expression type.

        template <typename ScalarType, typename FunctionType>
        using first_derivatives_value_t =
          typename WeakForms::internal::Differentiation::
            DiffOpResult<FunctionType, field_values_t<ScalarType>>::type;

        template <typename ScalarType, typename FunctionType>
        using second_derivatives_value_t =
          typename WeakForms::internal::Differentiation::DiffOpResult<
            first_derivatives_value_t<ScalarType, FunctionType>,
            field_values_t<ScalarType>>::type;

        // ========================
        // Generic helper functions
        // ========================

        static constexpr int
        n_operators()
        {
          return sizeof...(SymbolicOpsSubSpaceFieldSolution);
        }

        // ===================
        // AD helper functions
        // ===================

        static constexpr unsigned int
        get_n_components()
        {
          return unpack_n_components<SymbolicOpsSubSpaceFieldSolution...>();
        }

        static field_extractors_t
        get_initialized_extractors()
        {
          field_extractors_t field_extractors;
          unsigned int       n_previous_field_components = 0;

          unpack_initialize_extractors<0, SymbolicOpsSubSpaceFieldSolution...>(
            field_extractors, n_previous_field_components);

          return field_extractors;
        }

        template <typename SymbolicOpField>
        static typename SymbolicOpField::extractor_type
        get_initialized_extractor(const SymbolicOpField &field,
                                  const field_args_t &   field_args)
        {
          using Extractor_t = typename SymbolicOpField::extractor_type;
          unsigned int n_previous_field_components = 0;

          unpack_n_previous_field_components<
            0,
            SymbolicOpField,
            SymbolicOpsSubSpaceFieldSolution...>(field,
                                                 field_args,
                                                 n_previous_field_components);

          return Extractor_t(n_previous_field_components);
        }

        // =============
        // AD operations
        // =============

#ifdef DEAL_II_WITH_AUTO_DIFFERENTIATION

        template <typename ADHelperType, int dim, int spacedim>
        static void
        ad_register_independent_variables(
          ADHelperType &                          ad_helper,
          MeshWorker::ScratchData<dim, spacedim> &scratch_data,
          const std::vector<SolutionExtractionData<dim, spacedim>>
            &                       solution_extraction_data,
          const unsigned int        q_point,
          const field_args_t &      field_args,
          const field_extractors_t &field_extractors)
        {
          unpack_ad_register_independent_variables<
            0,
            ADHelperType,
            dim,
            spacedim,
            SymbolicOpsSubSpaceFieldSolution...>(ad_helper,
                                                 scratch_data,
                                                 solution_extraction_data,
                                                 q_point,
                                                 field_args,
                                                 field_extractors);
        }

        template <typename ADHelperType,
                  typename ADFunctionType,
                  int dim,
                  int spacedim>
        static auto
        ad_call_function(
          const ADHelperType &                    ad_helper,
          const ADFunctionType &                  ad_function,
          MeshWorker::ScratchData<dim, spacedim> &scratch_data,
          const std::vector<SolutionExtractionData<dim, spacedim>>
            &                       solution_extraction_data,
          const unsigned int        q_point,
          const field_extractors_t &field_extractors)
        {
          // https://riptutorial.com/cplusplus/example/26687/turn-a-std--tuple-t-----into-function-parameters
          return unpack_ad_call_function(
            ad_helper,
            ad_function,
            scratch_data,
            solution_extraction_data,
            q_point,
            field_extractors,
            std::make_index_sequence<
              std::tuple_size<field_extractors_t>::value>());
        }

#endif // DEAL_II_WITH_AUTO_DIFFERENTIATION

        // ===================
        // SD helper functions
        // ===================

#ifdef DEAL_II_WITH_SYMENGINE

        template <typename SDNumberType>
        static field_values_t<SDNumberType>
        get_symbolic_fields(const field_args_t &      field_args,
                            const SymbolicDecorations decorator)
        {
          return unpack_get_symbolic_fields<SDNumberType>(
            field_args,
            decorator,
            std::make_index_sequence<std::tuple_size<field_args_t>::value>());
        }

        template <typename SDNumberType>
        static Differentiation::SD::types::substitution_map
        sd_get_symbol_map(
          const field_values_t<SDNumberType> &symbolic_field_values)
        {
          return unpack_sd_get_symbol_map<SDNumberType>(
            symbolic_field_values,
            std::make_index_sequence<
              std::tuple_size<field_values_t<SDNumberType>>::value>());
        }

        template <typename SDNumberType, typename SDFunctionType>
        static auto
        sd_call_function(
          const SDFunctionType &              sd_function,
          const field_values_t<SDNumberType> &symbolic_field_values,
          const bool                          compute_hash = true)
        {
          return unpack_sd_call_function<SDNumberType>(
            sd_function,
            symbolic_field_values,
            compute_hash,
            std::make_index_sequence<
              std::tuple_size<field_values_t<SDNumberType>>::value>());
        }

        // Expect SDSubstitutionFunctionType to be a std::function
        template <typename SDNumberType, typename SDSubstitutionFunctionType>
        static Differentiation::SD::types::substitution_map
        sd_call_substitution_function(
          const SDSubstitutionFunctionType &  substitution_function,
          const field_values_t<SDNumberType> &symbolic_field_values)
        {
          if (substitution_function)
            return unpack_sd_call_substitution_function<SDNumberType>(
              substitution_function,
              symbolic_field_values,
              std::make_index_sequence<
                std::tuple_size<field_values_t<SDNumberType>>::value>());
          else
            return Differentiation::SD::types::substitution_map{};
        }

        // SDExpressionType can be an SD::Expression or a tensor of expressions.
        // Tuples of the former types are dealt with by the other variant.
        // Unfortunately it looks like we need to use the SFINAE idiom to help
        // the compiler, as it might try to implicitly convert these types
        // to tuples and get confused between the two functions.
        template <
          typename SDNumberType,
          typename SDExpressionType,
          typename = typename std::enable_if<
            !TemplateRestrictions::is_tuple<SDExpressionType>::value>::type>
        static first_derivatives_value_t<SDNumberType, SDExpressionType>
        sd_differentiate(
          const SDExpressionType &            sd_expression,
          const field_values_t<SDNumberType> &symbolic_field_values,
          const bool                          compute_hash = true)
        {
          return unpack_sd_differentiate<SDNumberType>(
            sd_expression,
            symbolic_field_values,
            compute_hash,
            std::make_index_sequence<
              std::tuple_size<field_values_t<SDNumberType>>::value>());
        }

        template <typename SDNumberType, typename... SDExpressionTypes>
        static std::tuple<
          first_derivatives_value_t<SDNumberType, SDExpressionTypes>...>
        sd_differentiate(
          const std::tuple<SDExpressionTypes...> &sd_expressions,
          const field_values_t<SDNumberType> &    symbolic_field_values,
          const bool                              compute_hash = true)
        {
          return unpack_sd_differentiate<SDNumberType>(
            sd_expressions,
            symbolic_field_values,
            compute_hash,
            std::make_index_sequence<
              std::tuple_size<std::tuple<SDExpressionTypes...>>::value>(),
            std::make_index_sequence<
              std::tuple_size<field_values_t<SDNumberType>>::value>());
        }

        template <typename SDExpressionType>
        static void
        sd_substitute(
          SDExpressionType &                                  sd_expression,
          const Differentiation::SD::types::substitution_map &substitution_map)
        {
          Differentiation::SD::substitute(sd_expression, substitution_map);
        }

        template <typename... SDExpressionTypes>
        static void
        sd_substitute(
          std::tuple<SDExpressionTypes...> &                  sd_expressions,
          const Differentiation::SD::types::substitution_map &substitution_map)
        {
          unpack_sd_substitute<0, SDExpressionTypes...>(sd_expressions,
                                                        substitution_map);
        }

        template <typename SDNumberType, typename SDExpressionType>
        static first_derivatives_value_t<SDNumberType, SDExpressionType>
        sd_substitute_and_differentiate(
          const SDExpressionType &                            sd_expression,
          const Differentiation::SD::types::substitution_map &substitution_map,
          const field_values_t<SDNumberType> &symbolic_field_values,
          const bool                          compute_hash = true)
        {
          if (substitution_map.size() > 0)
            {
              SDExpressionType sd_expression_subs{sd_expression};
              sd_substitute(sd_expression_subs, substitution_map);
              return sd_differentiate<SDNumberType>(sd_expression_subs,
                                                    symbolic_field_values,
                                                    compute_hash);
            }
          else
            return sd_differentiate<SDNumberType>(sd_expression,
                                                  symbolic_field_values,
                                                  compute_hash);
        }

        template <typename /*SDNumberType*/,
                  typename SDExpressionType,
                  typename BatchOptimizerType>
        static void
        sd_register_functions(BatchOptimizerType &    batch_optimizer,
                              const SDExpressionType &values,
                              const bool check_hash_computed = true)
        {
          if (check_hash_computed)
            assert_hash_computed(values);

          batch_optimizer.register_function(values);
        }

        template <typename SDNumberType,
                  typename SDExpressionType,
                  typename BatchOptimizerType>
        static void
        sd_register_functions(
          BatchOptimizerType &batch_optimizer,
          const first_derivatives_value_t<SDNumberType, SDExpressionType>
            &        derivatives,
          const bool check_hash_computed = true)
        {
          return unpack_sd_register_1st_order_functions<SDNumberType,
                                                        SDExpressionType>(
            batch_optimizer,
            derivatives,
            check_hash_computed,
            std::make_index_sequence<std::tuple_size<
              first_derivatives_value_t<SDNumberType,
                                        SDExpressionType>>::value>());
        }

        template <typename SDNumberType,
                  typename SDExpressionType,
                  typename BatchOptimizerType>
        static void
        sd_register_functions(
          BatchOptimizerType &batch_optimizer,
          const second_derivatives_value_t<SDNumberType, SDExpressionType>
            &        derivatives,
          const bool check_hash_computed = true)
        {
          return unpack_sd_register_2nd_order_functions<SDNumberType,
                                                        SDExpressionType>(
            batch_optimizer, derivatives, check_hash_computed);
        }

        template <typename T>
        static void
        sd_assert_hash_computed(const T &expressions)
        {
          assert_hash_computed(expressions);
        }

        template <typename SDNumberType,
                  typename ScalarType,
                  int dim,
                  int spacedim>
        static Differentiation::SD::types::substitution_map
        sd_get_substitution_map(
          MeshWorker::ScratchData<dim, spacedim> &scratch_data,
          const std::vector<SolutionExtractionData<dim, spacedim>>
            &                                 solution_extraction_data,
          const unsigned int                  q_point,
          const field_values_t<SDNumberType> &symbolic_field_values,
          const field_args_t &                field_args)
        {
          static_assert(std::tuple_size<field_values_t<SDNumberType>>::value ==
                          std::tuple_size<field_args_t>::value,
                        "Size mismatch");

          Differentiation::SD::types::substitution_map substitution_map;

          unpack_sd_add_to_substitution_map<SDNumberType, ScalarType>(
            substitution_map,
            scratch_data,
            solution_extraction_data,
            q_point,
            symbolic_field_values,
            field_args);

          return substitution_map;
        }

#endif // DEAL_II_WITH_SYMENGINE

      private:
        // ===================
        // AD helper functions
        // ===================

        template <typename SymbolicOpType>
        static constexpr unsigned int
        get_symbolic_op_field_n_components()
        {
          return internal::SpaceOpComponentInfo<SymbolicOpType>::n_components;

          // using ArbitraryType = double;
          // return FieldType<typename SymbolicOpType::template value_type<
          //   ArbitraryType>>::n_components;
        }

        // End point
        template <typename SymbolicOpType>
        static constexpr unsigned int
        unpack_n_components()
        {
          return get_symbolic_op_field_n_components<SymbolicOpType>();
        }

        template <typename SymbolicOpType, typename... OtherSymbolicOpTypes>
        static constexpr
          typename std::enable_if<(sizeof...(OtherSymbolicOpTypes) > 0),
                                  unsigned int>::type
          unpack_n_components()
        {
          return unpack_n_components<SymbolicOpType>() +
                 unpack_n_components<OtherSymbolicOpTypes...>();
        }

        template <std::size_t I = 0,
                  typename... SymbolicOpType,
                  typename... FieldExtractors>
          static typename std::enable_if <
          I<sizeof...(SymbolicOpType), void>::type
          unpack_initialize_extractors(
            std::tuple<FieldExtractors...> &field_extractors,
            unsigned int &                  n_previous_field_components)
        {
          using FEValuesExtractorType = decltype(std::get<I>(field_extractors));
          std::get<I>(field_extractors) =
            FEValuesExtractorType(n_previous_field_components);

          // Move on to the next field, noting that we've allocated a certain
          // number of components to this scalar/vector/tensor field.
          using SymbolicOp_t = typename std::decay<decltype(
            std::get<I>(std::declval<field_args_t>()))>::type;
          n_previous_field_components +=
            get_symbolic_op_field_n_components<SymbolicOp_t>();
          unpack_initialize_extractors<I + 1, SymbolicOpType...>(
            field_extractors, n_previous_field_components);
        }

        // End point
        template <std::size_t I = 0,
                  typename... SymbolicOpType,
                  typename... FieldExtractors>
        static
          typename std::enable_if<I == sizeof...(SymbolicOpType), void>::type
          unpack_initialize_extractors(
            std::tuple<FieldExtractors...> &field_extractors,
            unsigned int &                  n_previous_field_components)
        {
          (void)field_extractors;
          (void)n_previous_field_components;
        }

        template <std::size_t I = 0,
                  typename SymbolicOpField,
                  typename... SymbolicOpType,
                  typename... FieldArgs>
          static typename std::enable_if <
          I<sizeof...(SymbolicOpType), void>::type
          unpack_n_previous_field_components(
            const SymbolicOpField &         field,
            const std::tuple<FieldArgs...> &field_args,
            unsigned int &                  n_previous_field_components)
        {
          // Exit if we've found the entry in the tuple that matches the input
          // field. We can only do this through string matching, since multiple
          // fields might be using an op with the same signature.
          const SymbolicDecorations decorator;
          const auto &              listed_field = std::get<I>(field_args);
          if (listed_field.as_ascii(decorator) == field.as_ascii(decorator))
            return;

          // Move on to the next field, noting that we've allocated a certain
          // number of components to this scalar/vector/tensor field.
          using SymbolicOp_t =
            typename std::decay<decltype(listed_field)>::type;
          n_previous_field_components +=
            get_symbolic_op_field_n_components<SymbolicOp_t>();
          unpack_n_previous_field_components<I + 1,
                                             SymbolicOpField,
                                             SymbolicOpType...>(
            field, field_args, n_previous_field_components);
        }

        // End point
        template <std::size_t I = 0,
                  typename SymbolicOpField,
                  typename... SymbolicOpType,
                  typename... FieldArgs>
        static
          typename std::enable_if<I == sizeof...(SymbolicOpType), void>::type
          unpack_n_previous_field_components(
            const SymbolicOpField &         field,
            const std::tuple<FieldArgs...> &field_args,
            unsigned int &                  n_previous_field_components)
        {
          (void)field;
          (void)field_args;
          (void)n_previous_field_components;
          AssertThrow(false,
                      ExcMessage(
                        "Could not find SymbolicOp for the field solution."));
        }

        // =============
        // AD operations
        // =============

#ifdef DEAL_II_WITH_AUTO_DIFFERENTIATION

        template <std::size_t I = 0,
                  typename ADHelperType,
                  int dim,
                  int spacedim,
                  typename... SymbolicOpType,
                  typename... FieldExtractors>
          static typename std::enable_if <
          I<sizeof...(SymbolicOpType), void>::type
          unpack_ad_register_independent_variables(
            ADHelperType &                          ad_helper,
            MeshWorker::ScratchData<dim, spacedim> &scratch_data,
            const std::vector<SolutionExtractionData<dim, spacedim>>
              &                                   solution_extraction_data,
            const unsigned int                    q_point,
            const std::tuple<SymbolicOpType...> & symbolic_op_field_solutions,
            const std::tuple<FieldExtractors...> &field_extractors)
        {
          using scalar_type = typename ADHelperType::scalar_type;

          const auto &symbolic_op_field_solution =
            std::get<I>(symbolic_op_field_solutions);
          const auto &                          field_solutions =
            symbolic_op_field_solution.template operator()<scalar_type>(
              scratch_data,
              solution_extraction_data); // Cached solution at all QPs
          Assert(q_point < field_solutions.size(),
                 ExcIndexRange(q_point, 0, field_solutions.size()));
          const auto &field_solution  = field_solutions[q_point];
          const auto &field_extractor = std::get<I>(field_extractors);

          ad_helper.register_independent_variable(field_solution,
                                                  field_extractor);

          unpack_ad_register_independent_variables<I + 1,
                                                   ADHelperType,
                                                   dim,
                                                   spacedim,
                                                   SymbolicOpType...>(
            ad_helper,
            scratch_data,
            solution_extraction_data,
            q_point,
            symbolic_op_field_solutions,
            field_extractors);
        }

        // Get update flags from a unary op: End point
        template <std::size_t I = 0,
                  typename ADHelperType,
                  int dim,
                  int spacedim,
                  typename... SymbolicOpType,
                  typename... FieldExtractors>
        static
          typename std::enable_if<I == sizeof...(SymbolicOpType), void>::type
          unpack_ad_register_independent_variables(
            ADHelperType &                          ad_helper,
            MeshWorker::ScratchData<dim, spacedim> &scratch_data,
            const std::vector<SolutionExtractionData<dim, spacedim>>
              &                                   solution_extraction_data,
            const unsigned int                    q_point,
            const std::tuple<SymbolicOpType...> & symbolic_op_field_solution,
            const std::tuple<FieldExtractors...> &field_extractors)
        {
          // Do nothing
          (void)ad_helper;
          (void)scratch_data;
          (void)solution_extraction_data;
          (void)q_point;
          (void)symbolic_op_field_solution;
          (void)field_extractors;
        }

        template <typename ADHelperType,
                  typename ADFunctionType,
                  int dim,
                  int spacedim,
                  typename... FieldExtractors,
                  std::size_t... I>
        static auto
        unpack_ad_call_function(
          const ADHelperType &                    ad_helper,
          const ADFunctionType &                  ad_function,
          MeshWorker::ScratchData<dim, spacedim> &scratch_data,
          const std::vector<SolutionExtractionData<dim, spacedim>>
            &                                   solution_extraction_data,
          const unsigned int &                  q_point,
          const std::tuple<FieldExtractors...> &field_extractors,
          const std::index_sequence<I...>)
        {
          // https://riptutorial.com/cplusplus/example/26687/turn-a-std--tuple-t-----into-function-parameters
          return ad_function(scratch_data,
                             solution_extraction_data,
                             q_point,
                             ad_helper.get_sensitive_variables(
                               std::get<I>(field_extractors))...);
        }

#endif // DEAL_II_WITH_AUTO_DIFFERENTIATION

        // ===================
        // SD helper functions
        // ===================

#ifdef DEAL_II_WITH_SYMENGINE

        template <typename SDNumberType,
                  typename... FieldArgs,
                  std::size_t... I>
        static field_values_t<SDNumberType>
        unpack_get_symbolic_fields(const std::tuple<FieldArgs...> &field_args,
                                   const SymbolicDecorations       decorator,
                                   const std::index_sequence<I...>)
        {
          return {internal::make_symbolic<SDNumberType>(std::get<I>(field_args),
                                                        decorator)...};
        }


        template <typename SDNumberType, std::size_t... I>
        static Differentiation::SD::types::substitution_map
        unpack_sd_get_symbol_map(
          const field_values_t<SDNumberType> &symbolic_field_values,
          const std::index_sequence<I...>)
        {
          return Differentiation::SD::make_symbol_map(
            std::get<I>(symbolic_field_values)...);
        }

        template <typename SDNumberType,
                  typename SDFunctionType,
                  std::size_t... I>
        static auto
        unpack_sd_call_function(
          const SDFunctionType &              sd_function,
          const field_values_t<SDNumberType> &symbolic_field_values,
          const bool                          compute_hash,
          const std::index_sequence<I...>)
        {
          auto result = sd_function(std::get<I>(symbolic_field_values)...);

          if (compute_hash)
            compute_hash_in_place(result);

          return result;
        }

        // Expect SDSubstitutionFunctionType to be a std::function
        template <typename SDNumberType,
                  typename SDSubstitutionFunctionType,
                  std::size_t... I>
        static Differentiation::SD::types::substitution_map
        unpack_sd_call_substitution_function(
          const SDSubstitutionFunctionType &  substitution_function,
          const field_values_t<SDNumberType> &symbolic_field_values,
          const std::index_sequence<I...>)
        {
          Assert(substitution_function, ExcNotInitialized());
          return substitution_function(std::get<I>(symbolic_field_values)...);
        }

        template <typename SDNumberType,
                  typename SDExpressionType,
                  std::size_t... I>
        static first_derivatives_value_t<SDNumberType, SDExpressionType>
        unpack_sd_differentiate(
          const SDExpressionType &            sd_expression,
          const field_values_t<SDNumberType> &symbolic_field_values,
          const bool                          compute_hash,
          const std::index_sequence<I...>)
        {
          first_derivatives_value_t<SDNumberType, SDExpressionType> result = {
            Differentiation::SD::differentiate(
              sd_expression, std::get<I>(symbolic_field_values))...};

          if (compute_hash)
            compute_hash_in_place(result);

          return result;
        }

        template <typename SDNumberType,
                  typename... SDExpressionTypes,
                  std::size_t... I,
                  std::size_t... J>
        static std::tuple<
          first_derivatives_value_t<SDNumberType, SDExpressionTypes>...>
        unpack_sd_differentiate(
          const std::tuple<SDExpressionTypes...> &sd_expressions,
          const field_values_t<SDNumberType> &    symbolic_field_values,
          const bool                              compute_hash,
          const std::index_sequence<I...>,
          const std::index_sequence<J...> &seq_j)
        {
          // For a fixed row "I", expand all the derivatives of expression "I"
          // with respect to fields "J"
          return {
            unpack_sd_differentiate<SDNumberType>(std::get<I>(sd_expressions),
                                                  symbolic_field_values,
                                                  compute_hash,
                                                  seq_j)...};
        }

        template <std::size_t I = 0, typename... SDExpressionTypes>
          static typename std::enable_if <
          I<sizeof...(SDExpressionTypes), void>::type
          unpack_sd_substitute(
            std::tuple<SDExpressionTypes...> &sd_expressions,
            const Differentiation::SD::types::substitution_map
              &substitution_map)
        {
          sd_substitute(std::get<I>(sd_expressions), substitution_map);
          unpack_sd_substitute<I + 1, SDExpressionTypes...>(sd_expressions,
                                                            substitution_map);
        }

        template <std::size_t I = 0, typename... SDExpressionTypes>
        static
          typename std::enable_if<I == sizeof...(SDExpressionTypes), void>::type
          unpack_sd_substitute(
            std::tuple<SDExpressionTypes...> &sd_expressions,
            const Differentiation::SD::types::substitution_map
              &substitution_map)
        {
          // Do nothing
          (void)sd_expressions;
          (void)substitution_map;
        }

        static void
        compute_hash_in_place(Differentiation::SD::Expression &expression)
        {
          expression.get_value().hash();
        }

        template <int rank, int dim>
        static void
        compute_hash_in_place(Tensor<rank, dim, Differentiation::SD::Expression>
                                &tensor_of_expressions)
        {
          for (Differentiation::SD::Expression *e =
                 tensor_of_expressions.begin_raw();
               e != tensor_of_expressions.end_raw();
               ++e)
            {
              compute_hash_in_place(*e);
            }
        }

        template <int rank, int dim>
        static void
        compute_hash_in_place(
          SymmetricTensor<rank, dim, Differentiation::SD::Expression>
            &tensor_of_expressions)
        {
          for (Differentiation::SD::Expression *e =
                 tensor_of_expressions.begin_raw();
               e != tensor_of_expressions.end_raw();
               ++e)
            {
              compute_hash_in_place(*e);
            }
        }

        static void
        compute_hash_in_place(
          Differentiation::SD::types::substitution_map &substitution_map)
        {
          (void)substitution_map;
        }

        template <typename T, typename... Args>
        static void
        compute_hash_in_place(T &expression, Args &...other_expressions)
        {
          compute_hash_in_place(expression);
          compute_hash_in_place(other_expressions...);
        }

        template <typename... SDExpressions>
        static void
        compute_hash_in_place(std::tuple<SDExpressions...> &expressions)
        {
          unpack_compute_hash_in_place(
            expressions,
            std::make_index_sequence<
              std::tuple_size<std::tuple<SDExpressions...>>::value>());
        }

        template <typename... SDExpressions, std::size_t... I>
        static void
        unpack_compute_hash_in_place(std::tuple<SDExpressions...> &expressions,
                                     const std::index_sequence<I...>)
        {
          compute_hash_in_place(std::get<I>(expressions)...);
        }

        static void
        assert_hash_computed(const Differentiation::SD::Expression &expression)
        {
          (void)expression;

          // Assert(expression.is_hashed(),
          //        ExcMessage("Scalar expression has not been hashed."));
        }

        template <int rank, int dim>
        static void
        assert_hash_computed(
          const Tensor<rank, dim, Differentiation::SD::Expression>
            &tensor_of_expressions)
        {
          (void)tensor_of_expressions;

          // for (const Differentiation::SD::Expression *e =
          //        tensor_of_expressions.begin_raw();
          //      e != tensor_of_expressions.end_raw();
          //      ++e)
          //   {
          //     Assert(e->is_hashed(),
          //            ExcMessage(
          //              "Tensor element expression has not been hashed."));
          //   }
        }

        template <int rank, int dim>
        static void
        assert_hash_computed(
          const SymmetricTensor<rank, dim, Differentiation::SD::Expression>
            &tensor_of_expressions)
        {
          (void)tensor_of_expressions;

          // for (const Differentiation::SD::Expression *e =
          //        tensor_of_expressions.begin_raw();
          //      e != tensor_of_expressions.end_raw();
          //      ++e)
          //   {
          //     Assert(
          //       e->is_hashed(),
          //       ExcMessage(
          //         "SymmetricTensor element expression has not been
          //         hashed."));
          //   }
        }

        template <typename T, typename... Args>
        static void
        assert_hash_computed(const T &expression,
                             const Args &...other_expressions)
        {
          assert_hash_computed(expression);
          assert_hash_computed(other_expressions...);
        }

        template <typename... SDExpressions>
        static void
        assert_hash_computed(const std::tuple<SDExpressions...> &expressions)
        {
          unpack_assert_hash_computed(
            expressions,
            std::make_index_sequence<
              std::tuple_size<std::tuple<SDExpressions...>>::value>());
        }

        template <typename... SDExpressions, std::size_t... I>
        static void
        unpack_assert_hash_computed(
          const std::tuple<SDExpressions...> &expressions,
          const std::index_sequence<I...>)
        {
          assert_hash_computed(std::get<I>(expressions)...);
        }

        // Registration for first derivatives (stored in a single tuple)
        // Register a single expression
        template <typename /*SDNumberType*/,
                  typename /*SDExpressionType*/,
                  typename... SDExpressions,
                  typename BatchOptimizerType,
                  std::size_t... I>
        static typename std::enable_if<
          (TemplateRestrictions::are_not_tuples<SDExpressions...>::value) &&
          (sizeof...(I) == 1)>::type
        unpack_sd_register_1st_order_functions(
          BatchOptimizerType &                batch_optimizer,
          const std::tuple<SDExpressions...> &derivatives,
          const bool                          check_hash_computed,
          const std::index_sequence<I...>)
        {
          if (check_hash_computed)
            assert_hash_computed(std::get<I>(derivatives)...);

          batch_optimizer.register_function(std::get<I>(derivatives)...);
        }

        // Registration for first derivatives (stored in a single tuple)
        // Register multiple expressions simultaneously
        template <typename /*SDNumberType*/,
                  typename /*SDExpressionType*/,
                  typename... SDExpressions,
                  typename BatchOptimizerType,
                  std::size_t... I>
        static typename std::enable_if<
          (TemplateRestrictions::are_not_tuples<SDExpressions...>::value) &&
          (sizeof...(I) > 1)>::type
        unpack_sd_register_1st_order_functions(
          BatchOptimizerType &                batch_optimizer,
          const std::tuple<SDExpressions...> &derivatives,
          const bool                          check_hash_computed,
          const std::index_sequence<I...>)
        {
          if (check_hash_computed)
            assert_hash_computed(std::get<I>(derivatives)...);

          batch_optimizer.register_functions(std::get<I>(derivatives)...);
        }

        // Registration for higher-order derivatives
        template <typename SDNumberType,
                  typename SDExpressionType,
                  std::size_t I = 0,
                  typename BatchOptimizerType,
                  typename... Ts>
        static typename std::enable_if<(I < sizeof...(Ts)), void>::type
        unpack_sd_register_2nd_order_functions(
          BatchOptimizerType &     batch_optimizer,
          const std::tuple<Ts...> &higher_order_derivatives,
          const bool               check_hash_computed)
        {
          static_assert(TemplateRestrictions::are_tuples<Ts...>::value,
                        "Expected all inner objects to be tuples");

          // Filter through the outer tuple and dispatch the work to the
          // other function (specialized for some first derivative types).
          // Note: A recursive call to sd_register_functions(), in the hopes
          // that it would detect std::get<I>(higher_order_derivatives) as a
          // lower-order derivative, does not seem to work. It, for some reason,
          // calls the higher-order variant again and sends us into an infinite
          // loop. So don't do that!
          using InnerTupleType = typename std::decay<decltype(
            std::get<I>(higher_order_derivatives))>::type;
          unpack_sd_register_1st_order_functions<SDNumberType,
                                                 SDExpressionType>(
            batch_optimizer,
            std::get<I>(higher_order_derivatives),
            check_hash_computed,
            std::make_index_sequence<std::tuple_size<InnerTupleType>::value>());
          unpack_sd_register_2nd_order_functions<SDNumberType,
                                                 SDExpressionType,
                                                 I + 1>(
            batch_optimizer, higher_order_derivatives, check_hash_computed);
        }


        template <typename /*SDNumberType*/,
                  typename /*SDExpressionType*/,
                  std::size_t I = 0,
                  typename BatchOptimizerType,
                  typename... Ts>
        static typename std::enable_if<(I == sizeof...(Ts)), void>::type
        unpack_sd_register_2nd_order_functions(
          BatchOptimizerType &     batch_optimizer,
          const std::tuple<Ts...> &higher_order_derivatives,
          const bool               check_hash_computed)
        {
          // Do nothing
          (void)batch_optimizer;
          (void)higher_order_derivatives;
          (void)check_hash_computed;
        }

        template <typename SDNumberType,
                  typename ScalarType,
                  std::size_t I = 0,
                  int         dim,
                  int         spacedim,
                  typename... SymbolicOpType>
          static typename std::enable_if <
          I<sizeof...(SymbolicOpType), void>::type
          unpack_sd_add_to_substitution_map(
            Differentiation::SD::types::substitution_map &substitution_map,
            MeshWorker::ScratchData<dim, spacedim> &      scratch_data,
            const std::vector<SolutionExtractionData<dim, spacedim>>
              &                                  solution_extraction_data,
            const unsigned int                   q_point,
            const field_values_t<SDNumberType> & symbolic_field_values,
            const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions)
        {
          static_assert(std::tuple_size<field_values_t<SDNumberType>>::value ==
                          std::tuple_size<std::tuple<SymbolicOpType...>>::value,
                        "Size mismatch");

          // Get the field value
          const auto &symbolic_op_field_solution =
            std::get<I>(symbolic_op_field_solutions);
          const auto &                          field_solutions =
            symbolic_op_field_solution.template operator()<ScalarType>(
              scratch_data,
              solution_extraction_data); // Cached solution at all QPs
          Assert(q_point < field_solutions.size(),
                 ExcIndexRange(q_point, 0, field_solutions.size()));
          const auto &field_solution = field_solutions[q_point];

          // Get the symbol for the field
          const auto &symbolic_field_solution =
            std::get<I>(symbolic_field_values);

          // Append these to the substitution map, and recurse.
          Differentiation::SD::add_to_substitution_map(substitution_map,
                                                       symbolic_field_solution,
                                                       field_solution);
          unpack_sd_add_to_substitution_map<SDNumberType, ScalarType, I + 1>(
            substitution_map,
            scratch_data,
            solution_extraction_data,
            q_point,
            symbolic_field_values,
            symbolic_op_field_solutions);
        }

        template <typename SDNumberType,
                  typename ScalarType,
                  std::size_t I = 0,
                  int         dim,
                  int         spacedim,
                  typename... SymbolicOpType>
        static
          typename std::enable_if<I == sizeof...(SymbolicOpType), void>::type
          unpack_sd_add_to_substitution_map(
            Differentiation::SD::types::substitution_map &substitution_map,
            MeshWorker::ScratchData<dim, spacedim> &      scratch_data,
            const std::vector<SolutionExtractionData<dim, spacedim>>
              &                                  solution_extraction_data,
            const unsigned int                   q_point,
            const field_values_t<SDNumberType> & symbolic_field_values,
            const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions)
        {
          // Do nothing
          (void)substitution_map;
          (void)scratch_data;
          (void)solution_extraction_data;
          (void)q_point;
          (void)symbolic_field_values;
          (void)symbolic_op_field_solutions;
        }

#endif // DEAL_II_WITH_SYMENGINE

      }; // struct SymbolicOpsSubSpaceFieldSolutionHelper

    } // namespace internal
  }   // namespace Operators

} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE


#endif // dealii_weakforms_ad_sd_functor_internal_h
