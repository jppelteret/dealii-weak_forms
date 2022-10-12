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

#ifndef dealii_weakforms_functors_h
#define dealii_weakforms_functors_h

#include <deal.II/base/config.h>

#include <deal.II/base/function.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <weak_forms/config.h>
#include <weak_forms/numbers.h>
#include <weak_forms/sd_expression_internal.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/types.h>


WEAK_FORMS_NAMESPACE_OPEN


#ifndef DOXYGEN

// Forward declarations
namespace WeakForms
{
  template <int rank>
  class Functor;

  class ScalarFunctor;

  template <int rank, int spacedim>
  class TensorFunctor;

  template <int rank, int spacedim>
  class SymmetricTensorFunctor;

  template <int spacedim>
  class ScalarFunctionFunctor;

  template <int rank, int spacedim>
  class TensorFunctionFunctor;
} // namespace WeakForms

#endif // DOXYGEN


namespace WeakForms
{
  /**
   * @brief A base class for other objects that represent functors.
   *
   * Functors are classes that act like a function. In this context, these
   * functors are to be able to return a value of specific rank at any point in
   * space (be it in a bulk material, on a body's boundary, or on an interface).
   *
   * @tparam rank_ The (tensorial) rank of the return type that this
   * object computes.
   *
   * \ingroup functors
   */
  template <int rank_>
  class Functor
  {
  public:
    /**
     * Rank of this object operates.
     */
    static const unsigned int rank = rank_;

    Functor(const std::string &symbol_ascii, const std::string &symbol_latex)
      : symbol_ascii(symbol_ascii)
      , symbol_latex(symbol_latex != "" ? symbol_latex : symbol_ascii)
    {}

    virtual ~Functor() = default;

    // ----  Ascii ----

    virtual std::string
    as_ascii(const SymbolicDecorations &decorator) const
    {
      return decorator.symbolic_op_functor_as_ascii(*this, rank);
    }

    virtual std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const
    {
      (void)decorator;
      return symbol_ascii;
    }

    // ---- LaTeX ----

    virtual std::string
    as_latex(const SymbolicDecorations &decorator) const
    {
      return decorator.symbolic_op_functor_as_latex(*this, rank);
    }

    virtual std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const
    {
      (void)decorator;
      return symbol_latex;
    }

  protected:
    const std::string symbol_ascii;
    const std::string symbol_latex;
  };



  /**
   * @brief A functor that returns scalar values upon evaluation.
   *
   * This class wraps generic functions that can be evaluated at any numerical
   * quadrature point. The output of these functions are of scalar type.
   * Since the definition of the evaluation call is provided through a
   * `std::function`, the computation pipeline can be made dependent on user
   * data if it is passed in through the capture clause of a lambda function.
   *
   * An example of usage:
   * @code {.cpp}
   * const ScalarFunctor c1("c1", "c1");
   *
   * const auto f1 = c1.template value<double, dim, spacedim>(
   *   [](const FEValuesBase<dim, spacedim> &, const unsigned int)
   *   { return 1.0; });
   * @endcode
   *
   * \ingroup functors
   */
  class ScalarFunctor : public Functor<0>
  {
    using Base = Functor<0>;

  public:
    template <typename ScalarType>
    using value_type = ScalarType;

    template <typename ScalarType, int dim, int spacedim = dim>
    using function_type = std::function<
      value_type<ScalarType>(const FEValuesBase<dim, spacedim> &fe_values,
                             const unsigned int                 q_point)>;

    template <typename ScalarType, int dim, int spacedim = dim>
    using interface_function_type = std::function<value_type<ScalarType>(
      const FEInterfaceValues<dim, spacedim> &fe_interface_values,
      const unsigned int                      q_point)>;

    ScalarFunctor(const std::string &symbol_ascii,
                  const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    // Methods to promote this class to a SymbolicOp
    template <typename ScalarType, int dim, int spacedim = dim>
    auto
    value(const function_type<ScalarType, dim, spacedim> &function,
          const interface_function_type<ScalarType, dim, spacedim>
            &               interface_function,
          const UpdateFlags update_flags = update_default) const;

    template <typename ScalarType, int dim, int spacedim = dim>
    auto
    value(const function_type<ScalarType, dim, spacedim> &function,
          const UpdateFlags update_flags = update_default) const
    {
      const interface_function_type<ScalarType, dim, spacedim> dummy_function;
      return this->value(function, dummy_function, update_flags);
    }

    template <typename ScalarType, int dim, int spacedim = dim>
    auto
    value(const interface_function_type<ScalarType, dim, spacedim>
            &               interface_function,
          const UpdateFlags update_flags = update_default) const
    {
      const function_type<ScalarType, dim, spacedim> dummy_function;
      return this->value(dummy_function, interface_function, update_flags);
    }
  };



  /**
   * @brief A functor that returns a tensor upon evaluation.
   *
   * This class wraps generic functions that can be evaluated at any numerical
   * quadrature point. The output of these functions are a tensorial type of
   * the given @p rank.
   * Since the definition of the evaluation call is provided through a
   * `std::function`, the computation pipeline can be made dependent on user
   * data if it is passed in through the capture clause of a lambda function.
   *
   * An example of usage:
   * @code {.cpp}
   * const TensorFunctor<2, spacedim> t1("C1", "C1");
   *
   * const auto tf1 = t1.template value<double, spacedim>(
   *   [](const FEValuesBase<dim, spacedim> &, const unsigned int)
   *   { return Tensor<2, dim, double>(unit_symmetric_tensor<spacedim>()); });
   * @endcode
   *
   * @tparam rank The rank of the tensor that is returned upon evaluation.
   * @tparam spacedim The spatial dimension of the tensor that is returned
   * upon evaluation.
   *
   * \ingroup functors
   */
  template <int rank, int spacedim>
  class TensorFunctor : public Functor<rank>
  {
    using Base = Functor<rank>;

  public:
    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = spacedim;

    template <typename ScalarType>
    using value_type = Tensor<rank, spacedim, ScalarType>;

    template <typename ScalarType, int dim = spacedim>
    using function_type = std::function<
      value_type<ScalarType>(const FEValuesBase<dim, spacedim> &fe_values,
                             const unsigned int                 q_point)>;

    template <typename ScalarType, int dim = spacedim>
    using interface_function_type = std::function<value_type<ScalarType>(
      const FEInterfaceValues<dim, spacedim> &fe_interface_values,
      const unsigned int                      q_point)>;

    TensorFunctor(const std::string &symbol_ascii,
                  const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    // Methods to promote this class to a SymbolicOp
    template <typename ScalarType, int dim = spacedim>
    auto
    value(const function_type<ScalarType, dim> &          function,
          const interface_function_type<ScalarType, dim> &interface_function,
          const UpdateFlags update_flags = update_default) const;

    template <typename ScalarType, int dim = spacedim>
    auto
    value(const function_type<ScalarType, dim> &function,
          const UpdateFlags update_flags = update_default) const
    {
      const interface_function_type<ScalarType, dim> dummy_function;
      return this->value(function, dummy_function, update_flags);
    }

    template <typename ScalarType, int dim = spacedim>
    auto
    value(const interface_function_type<ScalarType, dim> &interface_function,
          const UpdateFlags update_flags = update_default) const
    {
      const function_type<ScalarType, dim> dummy_function;
      return this->value(dummy_function, interface_function, update_flags);
    }
  };



  /**
   * @brief A functor that returns a symmetric tensor upon evaluation.
   *
   * This class wraps generic functions that can be evaluated at any numerical
   * quadrature point. The output of these functions are a (symmetric) tensorial
   * type of the given @p rank.
   * Since the definition of the evaluation call is provided through a
   * `std::function`, the computation pipeline can be made dependent on user
   * data if it is passed in through the capture clause of a lambda function.
   *
   * An example of usage:
   * @code {.cpp}
   * const SymmetricTensorFunctor<2, spacedim> s1("S1", "S1");
   *
   * const auto sf1 = s1.template value<double, spacedim>(
   *   [](const FEValuesBase<dim, spacedim> &, const unsigned int)
   *   { return unit_symmetric_tensor<spacedim>(); });
   * @endcode
   *
   * @tparam rank The rank of the tensor that is returned upon evaluation.
   * @tparam spacedim The spatial dimension of the tensor that is returned
   * upon evaluation.
   *
   * \ingroup functors
   */
  template <int rank, int spacedim>
  class SymmetricTensorFunctor : public Functor<rank>
  {
    static_assert(rank == 2 || rank == 4, "Invalid rank");
    using Base = Functor<rank>;

  public:
    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = spacedim;

    template <typename ScalarType>
    using value_type = SymmetricTensor<rank, spacedim, ScalarType>;

    template <typename ScalarType, int dim = spacedim>
    using function_type = std::function<
      value_type<ScalarType>(const FEValuesBase<dim, spacedim> &fe_values,
                             const unsigned int                 q_point)>;

    template <typename ScalarType, int dim = spacedim>
    using interface_function_type = std::function<value_type<ScalarType>(
      const FEInterfaceValues<dim, spacedim> &fe_interface_values,
      const unsigned int                      q_point)>;

    SymmetricTensorFunctor(const std::string &symbol_ascii,
                           const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    // Methods to promote this class to a SymbolicOp
    template <typename ScalarType, int dim = spacedim>
    auto
    value(const function_type<ScalarType, dim> &          function,
          const interface_function_type<ScalarType, dim> &interface_function,
          const UpdateFlags update_flags = update_default) const;

    template <typename ScalarType, int dim = spacedim>
    auto
    value(const function_type<ScalarType, dim> &function,
          const UpdateFlags update_flags = update_default) const
    {
      const interface_function_type<ScalarType, dim> dummy_function;
      return this->value(function, dummy_function, update_flags);
    }

    template <typename ScalarType, int dim = spacedim>
    auto
    value(const interface_function_type<ScalarType, dim> &interface_function,
          const UpdateFlags update_flags = update_default) const
    {
      const function_type<ScalarType, dim> dummy_function;
      return this->value(dummy_function, interface_function, update_flags);
    }
  };



  /**
   * @brief A functor that acts as a wrapper for a scalar deal.II function.
   *
   * This class wraps deal.II and user functions that are derived from a
   * <a
   * href="https://www.dealii.org/current/doxygen/deal.II/classFunction.html">`dealii::Function`</a>.
   * As such, they are able to provide both the value and gradients of functions
   * that implement these methods.
   *
   * An example of usage:
   * @code {.cpp}
   * const ScalarFunctionFunctor<spacedim> c1("c1", "c1");
   * const Functions::ConstantFunction<spacedim, double>
   *   constant_scalar_function_1(1.0);
   *
   * // Function value
   * const auto f1 = c1.template value<double, dim>(constant_scalar_function_1);
   *
   * // Function gradient
   * const auto gf1 = c1.template gradient<double, dim>(
   *   constant_scalar_function_1);
   * @endcode
   *
   * @tparam spacedim The spatial dimension in which this function is being
   * evaluated. (Although its not vital information for the value call itself,
   * this is required in order to compute the gradient of the function.)
   *
   * \ingroup functors
   */
  template <int spacedim>
  class ScalarFunctionFunctor : public Functor<0>
  {
    using Base = Functor<0>;

  public:
    template <typename ScalarType>
    using function_type = Function<spacedim, ScalarType>;

    template <typename ScalarType>
    using value_type = ScalarType;

    template <typename ScalarType>
    using gradient_type = Tensor<1, spacedim, ScalarType>;

    ScalarFunctionFunctor(const std::string &symbol_ascii,
                          const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    virtual std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const
    {
      return decorator.make_position_dependent_symbol_ascii(this->symbol_ascii);
    }

    virtual std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const
    {
      return decorator.make_position_dependent_symbol_latex(this->symbol_latex);
    }

    template <typename ScalarType, int dim = spacedim>
    auto
    value(const function_type<ScalarType> &function,
          const UpdateFlags                update_flags = update_default) const;

    template <typename ScalarType, int dim = spacedim>
    auto
    gradient(const function_type<ScalarType> &function,
             const UpdateFlags update_flags = update_default) const;
  };



  /**
   * @brief A functor that acts as a wrapper for a tensor-valued deal.II function.
   *
   * This class wraps deal.II and user functions that are derived from a
   * <a
   * href="https://www.dealii.org/current/doxygen/deal.II/classTensorFunction.html">`dealii::TensorFunction`</a>.
   * As such, they are able to provide both the value and gradients of functions
   * that implement these methods.
   *
   * An example of usage:
   * @code {.cpp}
   * const TensorFunctionFunctor<2, spacedim>     t1("C1", "C1");
   * const ConstantTensorFunction<2, dim, double> constant_tensor_function_1(
   *   unit_symmetric_tensor<dim>());
   *
   * // Function value
   * const auto tf1 = t1.template value<double,
   *                                    dim>(constant_tensor_function_1);
   *
   * // Function gradient
   * const auto gtf1 = t1.template gradient<double,
   *                                        dim>(constant_tensor_function_1);
   * @endcode
   *
   * @tparam rank The rank of the tensor that is returned upon evaluation.
   * @tparam spacedim The spatial dimension of the tensor that is returned
   * upon evaluation.
   *
   * \ingroup functors
   */
  template <int rank, int spacedim>
  class TensorFunctionFunctor : public Functor<rank>
  {
    using Base = Functor<rank>;

  public:
    /**
     * Dimension in which this object operates.
     */
    static const unsigned int dimension = spacedim;

    template <typename ScalarType>
    using function_type = TensorFunction<rank, spacedim, ScalarType>;

    template <typename ScalarType>
    using value_type = typename function_type<ScalarType>::value_type;

    template <typename ResultScalarType>
    using gradient_type =
      typename function_type<ResultScalarType>::gradient_type;

    TensorFunctionFunctor(const std::string &symbol_ascii,
                          const std::string &symbol_latex)
      : Base(symbol_ascii, symbol_latex)
    {}

    virtual std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const
    {
      return decorator.make_position_dependent_symbol_ascii(this->symbol_ascii);
    }

    virtual std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const
    {
      return decorator.make_position_dependent_symbol_latex(this->symbol_latex);
    }

    template <typename ScalarType, int dim = spacedim>
    auto
    value(const function_type<ScalarType> &function,
          const UpdateFlags                update_flags = update_default) const;

    template <typename ScalarType, int dim = spacedim>
    auto
    gradient(const function_type<ScalarType> &function,
             const UpdateFlags update_flags = update_default) const;
  };



  /**
   * An alias for a functor that returns vector (i.e. rank-1 tensor) values
   * upon evaluation.
   *
   * An example of usage:
   * @code {.cpp}
   * const VectorFunctor<spacedim> v1("v1", "v1");
   *
   * const auto vf1 = v1.template value<double, spacedim>(
   *   [](const FEValuesBase<dim, spacedim> &, const unsigned int)
   *   { return Tensor<1, dim, double>{}; });
   * @endcode
   *
   * \ingroup functors
   */
  template <int dim>
  using VectorFunctor = TensorFunctor<1, dim>;



  /**
   * An alias for a functor that wraps a rank-1 deal.II tensor function.
   *
   * An example of usage:
   * @code {.cpp}
   * const VectorFunctionFunctor<spacedim> v1("v1", "v1");
   * const ConstantTensorFunction<1, dim, double> constant_tensor_function_1(
   *   Tensor<1, dim, double>{});
   *
   * // Function value
   * const auto vf1 = v1.template value<double,
   *                                    dim>(constant_tensor_function_1);
   *
   * // Function gradient
   * const auto gvf1 = v1.template gradient<double,
   *                                        dim>(constant_tensor_function_1);
   * @endcode
   *
   * \ingroup functors
   */
  template <int dim>
  using VectorFunctionFunctor = TensorFunctionFunctor<1, dim>;

} // namespace WeakForms



/* ================== Specialization of unary operators ================== */



namespace WeakForms
{
  namespace internal
  {
    // Used to work around the restriction that template arguments
    // for template type parameter must be a type
    template <int dim_, int spacedim_>
    struct DimPack
    {
      static const unsigned int dim      = dim_;
      static const unsigned int spacedim = spacedim_;
    };
  } // namespace internal

  namespace Operators
  {
#ifdef DEAL_II_WITH_SYMENGINE

/**
 * A macro that performs a conversion of the functor to a symbolic
 * expression type.
 */
#  define DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_COMMON_IMPL()                 \
    value_type<dealii::Differentiation::SD::Expression> as_expression(         \
      const SymbolicDecorations &decorator = SymbolicDecorations()) const      \
    {                                                                          \
      return WeakForms::Operators::internal::make_symbolic<                    \
        value_type<dealii::Differentiation::SD::Expression>>(                  \
        this->as_ascii(decorator));                                            \
    }                                                                          \
                                                                               \
    Differentiation::SD::types::substitution_map get_symbol_registration_map() \
      const                                                                    \
    {                                                                          \
      return Differentiation::SD::make_symbol_map(this->as_expression());      \
    }                                                                          \
                                                                               \
    Differentiation::SD::types::substitution_map                               \
    get_intermediate_substitution_map() const                                  \
    {                                                                          \
      return Differentiation::SD::types::substitution_map{};                   \
    }

#  define DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_FUNCTOR_IMPL()        \
    DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_COMMON_IMPL()               \
                                                                       \
    Differentiation::SD::types::substitution_map get_substitution_map( \
      const MeshWorker::ScratchData<dim, spacedim> &scratch_data,      \
      const std::vector<SolutionExtractionData<dim, spacedim>>         \
        &                solution_extraction_data,                     \
      const unsigned int q_point) const                                \
    {                                                                  \
      (void)solution_extraction_data;                                  \
      const auto &fe_values = scratch_data.get_current_fe_values();    \
      using Result_t        = decltype(function(fe_values, q_point));  \
      return Differentiation::SD::make_substitution_map(               \
        this->as_expression(),                                         \
        this->template operator()<Result_t>(fe_values, q_point));      \
    }

#  define DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_FUNCTION_FUNCTOR_IMPL()    \
    DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_COMMON_IMPL()                    \
                                                                            \
    Differentiation::SD::types::substitution_map get_substitution_map(      \
      const MeshWorker::ScratchData<dim, spacedim> &scratch_data,           \
      const std::vector<SolutionExtractionData<dim, spacedim>>              \
        &                solution_extraction_data,                          \
      const unsigned int q_point) const                                     \
    {                                                                       \
      (void)scratch_data;                                                   \
      (void)solution_extraction_data;                                       \
      (void)q_point;                                                        \
      const auto &point = scratch_data.get_quadrature_points()[q_point];    \
      using Result_t    = decltype(function->value(point));                 \
      return Differentiation::SD::make_substitution_map(                    \
        this->as_expression(), this->template operator()<Result_t>(point)); \
    }


#else // DEAL_II_WITH_SYMENGINE

/**
 * A dummy macro.
 */
#  define DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_COMMON_IMPL() ;
#  define DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_FUNCTOR_IMPL() ;
#  define DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_FUNCTION_FUNCTOR_IMPL() ;

#endif // DEAL_II_WITH_SYMENGINE


    /* ------------------------ Functors: Custom ------------------------ */


#define DEAL_II_SYMBOLIC_OP_FUNCTOR_COMMON_IMPL()                             \
public:                                                                       \
  /**                                                                         \
   * Dimension in which this object operates.                                 \
   */                                                                         \
  static const unsigned int dimension = dim;                                  \
                                                                              \
  /**                                                                         \
   * Dimension of the space in which this object operates.                    \
   */                                                                         \
  static const unsigned int space_dimension = spacedim;                       \
                                                                              \
  using scalar_type = ScalarType;                                             \
                                                                              \
  template <typename ResultScalarType>                                        \
  using return_type = std::vector<value_type<ResultScalarType>>;              \
                                                                              \
  template <typename ResultScalarType, std::size_t width>                     \
  using vectorized_value_type = typename numbers::VectorizedValue<            \
    value_type<ResultScalarType>>::template type<width>;                      \
                                                                              \
  template <typename ResultScalarType, std::size_t width>                     \
  using vectorized_return_type = typename numbers::VectorizedValue<           \
    value_type<ResultScalarType>>::template type<width>;                      \
                                                                              \
  static const enum SymbolicOpCodes op_code = SymbolicOpCodes::value;         \
                                                                              \
  explicit SymbolicOp(                                                        \
    const Op &                                 operand,                       \
    const function_type<ScalarType> &          function,                      \
    const interface_function_type<ScalarType> &interface_function,            \
    const UpdateFlags                          update_flags)                  \
    : operand(operand)                                                        \
    , function(function)                                                      \
    , interface_function(interface_function)                                  \
    , update_flags(update_flags)                                              \
  {}                                                                          \
                                                                              \
  explicit SymbolicOp(const Op &operand, const value_type<ScalarType> &value) \
    : SymbolicOp(                                                             \
        operand,                                                              \
        [value](const FEValuesBase<dim, spacedim> &, const unsigned int)      \
        { return value; },                                                    \
        [value](const FEInterfaceValues<dim, spacedim> &, const unsigned int) \
        { return value; })                                                    \
  {}                                                                          \
                                                                              \
  std::string as_ascii(const SymbolicDecorations &decorator) const            \
  {                                                                           \
    const auto &naming = decorator.get_naming_ascii().differential_operators; \
    return decorator.decorate_with_operator_ascii(                            \
      naming.value, operand.as_ascii(decorator));                             \
  }                                                                           \
                                                                              \
  std::string as_latex(const SymbolicDecorations &decorator) const            \
  {                                                                           \
    const auto &naming = decorator.get_naming_latex().differential_operators; \
    return decorator.decorate_with_operator_latex(                            \
      naming.value, operand.as_latex(decorator));                             \
  }                                                                           \
                                                                              \
  UpdateFlags get_update_flags() const                                        \
  {                                                                           \
    return update_flags;                                                      \
  }                                                                           \
                                                                              \
  /**                                                                         \
   * Return values at all quadrature points                                   \
   *                                                                          \
   * This is generic enough that it can operate on cells, faces and           \
   * subfaces. The user can cast the @p fe_values values object into          \
   * the base type for face values if necessary. The user can get the         \
   * current cell by a call to `fe_values.get_cell()` and, if cast to         \
   * an FEFaceValuesBase type, then `fe_face_values.get_face_index()`         \
   * returns the face index.                                                  \
   */                                                                         \
  template <typename ResultScalarType>                                        \
  return_type<ResultScalarType> operator()(                                   \
    const FEValuesBase<dim, spacedim> &fe_values) const                       \
  {                                                                           \
    return_type<ResultScalarType> out;                                        \
    out.reserve(fe_values.n_quadrature_points);                               \
                                                                              \
    for (const auto q_point : fe_values.quadrature_point_indices())           \
      out.emplace_back(                                                       \
        this->template operator()<ResultScalarType>(fe_values, q_point));     \
                                                                              \
    return out;                                                               \
  }                                                                           \
                                                                              \
  template <typename ResultScalarType>                                        \
  return_type<ResultScalarType> operator()(                                   \
    const FEInterfaceValues<dim, spacedim> &fe_interface_values) const        \
  {                                                                           \
    return_type<ResultScalarType> out;                                        \
    out.reserve(fe_interface_values.n_quadrature_points);                     \
                                                                              \
    for (const auto q_point : fe_interface_values.quadrature_point_indices()) \
      out.emplace_back(                                                       \
        this->template operator()<ResultScalarType>(fe_interface_values,      \
                                                    q_point));                \
                                                                              \
    return out;                                                               \
  }                                                                           \
                                                                              \
  /**                                                                         \
   * Return a vectorized set of values for a given quadrature point range.    \
   */                                                                         \
  template <typename ResultScalarType, std::size_t width>                     \
  vectorized_return_type<ResultScalarType, width> operator()(                 \
    const FEValuesBase<dim, spacedim> & fe_values,                            \
    const types::vectorized_qp_range_t &q_point_range) const                  \
  {                                                                           \
    vectorized_return_type<ResultScalarType, width> out;                      \
    Assert(q_point_range.size() <= width,                                     \
           ExcIndexRange(q_point_range.size(), 0, width));                    \
                                                                              \
    /* TODO: Can we guarantee that the underlying function is immutable? */   \
    DEAL_II_OPENMP_SIMD_PRAGMA                                                \
    for (unsigned int i = 0; i < q_point_range.size(); ++i)                   \
      if (q_point_range[i] < fe_values.n_quadrature_points)                   \
        numbers::set_vectorized_values(                                       \
          out,                                                                \
          i,                                                                  \
          this->template operator()<ResultScalarType>(fe_values,              \
                                                      q_point_range[i]));     \
                                                                              \
    return out;                                                               \
  }                                                                           \
                                                                              \
  template <typename ResultScalarType, std::size_t width>                     \
  vectorized_return_type<ResultScalarType, width> operator()(                 \
    const FEInterfaceValues<dim, spacedim> &fe_interface_values,              \
    const types::vectorized_qp_range_t &    q_point_range) const                  \
  {                                                                           \
    vectorized_return_type<ResultScalarType, width> out;                      \
    Assert(q_point_range.size() <= width,                                     \
           ExcIndexRange(q_point_range.size(), 0, width));                    \
                                                                              \
    /* TODO: Can we guarantee that the underlying function is immutable? */   \
    DEAL_II_OPENMP_SIMD_PRAGMA                                                \
    for (unsigned int i = 0; i < q_point_range.size(); ++i)                   \
      if (q_point_range[i] < fe_interface_values.n_quadrature_points)         \
        numbers::set_vectorized_values(                                       \
          out,                                                                \
          i,                                                                  \
          this->template operator()<ResultScalarType>(fe_interface_values,    \
                                                      q_point_range[i]));     \
                                                                              \
    return out;                                                               \
  }                                                                           \
                                                                              \
  DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_FUNCTOR_IMPL()                       \
                                                                              \
private:                                                                      \
  const Op                                  operand;                          \
  const function_type<ScalarType>           function;                         \
  const interface_function_type<ScalarType> interface_function;               \
  const UpdateFlags                         update_flags;                     \
                                                                              \
  template <typename ResultScalarType>                                        \
  value_type<ResultScalarType> operator()(                                    \
    const FEValuesBase<dim, spacedim> &fe_values, const unsigned int q_point) \
    const                                                                     \
  {                                                                           \
    Assert(function,                                                          \
           ExcMessage(                                                        \
             "Function not initialized for use on cells or boundaries."));    \
    return function(fe_values, q_point);                                      \
  }                                                                           \
                                                                              \
  template <typename ResultScalarType>                                        \
  value_type<ResultScalarType> operator()(                                    \
    const FEInterfaceValues<dim, spacedim> &fe_interface_values,              \
    const unsigned int                      q_point) const                                         \
  {                                                                           \
    Assert(interface_function,                                                \
           ExcMessage("Function not initialized for use on interfaces."));    \
    return interface_function(fe_interface_values, q_point);                  \
  }



    /**
     * Extract the value from a scalar functor.
     */
    template <typename ScalarType, int dim, int spacedim>
    class SymbolicOp<ScalarFunctor,
                     SymbolicOpCodes::value,
                     ScalarType,
                     WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = ScalarFunctor;

      template <typename ResultScalarType>
      using function_type =
        typename Op::template function_type<ResultScalarType, dim, spacedim>;

      template <typename ResultScalarType>
      using interface_function_type = typename Op::
        template interface_function_type<ResultScalarType, dim, spacedim>;

    public:
      template <typename ResultScalarType>
      using value_type = Op::template value_type<ResultScalarType>;

      DEAL_II_SYMBOLIC_OP_FUNCTOR_COMMON_IMPL()

    public:
      static const int rank = 0;
    };



    /**
     * Extract the value from a tensor functor.
     */
    template <typename ScalarType, int dim, int rank_, int spacedim>
    class SymbolicOp<TensorFunctor<rank_, spacedim>,
                     SymbolicOpCodes::value,
                     ScalarType,
                     WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = TensorFunctor<rank_, spacedim>;

      template <typename ResultScalarType>
      using function_type =
        typename Op::template function_type<ResultScalarType, dim>;

      template <typename ResultScalarType>
      using interface_function_type =
        typename Op::template interface_function_type<ResultScalarType, dim>;

    public:
      template <typename ResultScalarType>
      using value_type = typename Op::template value_type<ResultScalarType>;

      DEAL_II_SYMBOLIC_OP_FUNCTOR_COMMON_IMPL()

    public:
      static const int rank = rank_;
      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");
    };



    /**
     * Extract the value from a symmetric tensor functor.
     */
    template <typename ScalarType, int dim, int rank_, int spacedim>
    class SymbolicOp<SymmetricTensorFunctor<rank_, spacedim>,
                     SymbolicOpCodes::value,
                     ScalarType,
                     WeakForms::internal::DimPack<dim, spacedim>>
    {
      static_assert(rank_ == 2 || rank_ == 4, "Invalid rank");

      using Op = SymmetricTensorFunctor<rank_, spacedim>;

      template <typename ResultScalarType>
      using function_type =
        typename Op::template function_type<ResultScalarType, dim>;

      template <typename ResultScalarType>
      using interface_function_type =
        typename Op::template interface_function_type<ResultScalarType, dim>;

    public:
      template <typename ResultScalarType>
      using value_type = typename Op::template value_type<ResultScalarType>;

      DEAL_II_SYMBOLIC_OP_FUNCTOR_COMMON_IMPL()

    public:
      static const int rank = rank_;
      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");
    };


#undef DEAL_II_SYMBOLIC_OP_FUNCTOR_COMMON_IMPL



    /* ------------------------ Functors: deal.II ------------------------ */


#define DEAL_II_SYMBOLIC_OP_FUNCTION_FUNCTOR_COMMON_IMPL()                     \
public:                                                                        \
  /**                                                                          \
   * Dimension in which this object operates.                                  \
   */                                                                          \
  static const unsigned int dimension = dim;                                   \
                                                                               \
  /**                                                                          \
   * Dimension of the space in which this object operates.                     \
   */                                                                          \
  static const unsigned int space_dimension = spacedim;                        \
                                                                               \
  using scalar_type = ScalarType;                                              \
                                                                               \
  template <typename ResultScalarType>                                         \
  using function_type = typename Op::template function_type<ResultScalarType>; \
                                                                               \
  template <typename ResultScalarType>                                         \
  using return_type = std::vector<value_type<ResultScalarType>>;               \
                                                                               \
  template <typename ResultScalarType, std::size_t width>                      \
  using vectorized_value_type = typename numbers::VectorizedValue<             \
    value_type<ResultScalarType>>::template type<width>;                       \
                                                                               \
  template <typename ResultScalarType, std::size_t width>                      \
  using vectorized_return_type = typename numbers::VectorizedValue<            \
    value_type<ResultScalarType>>::template type<width>;                       \
                                                                               \
  static const enum SymbolicOpCodes op_code = SymbolicOpCodes::value;          \
                                                                               \
  /**                                                                          \
   * Construct a new Unary Op object                                           \
   *                                                                           \
   * @param operand                                                            \
   * @param function Non-owning, so the passed in @p function_type must have   \
   * a longer lifetime than this object.                                       \
   */                                                                          \
  explicit SymbolicOp(const Op &                       operand,                \
                      const function_type<ScalarType> &function,               \
                      const UpdateFlags &              update_flags)           \
    : operand(operand)                                                         \
    , function(&function)                                                      \
    , update_flags(update_flags)                                               \
  {}                                                                           \
                                                                               \
  UpdateFlags get_update_flags() const                                         \
  {                                                                            \
    return update_flags | UpdateFlags::update_quadrature_points;               \
  }                                                                            \
                                                                               \
  /**                                                                          \
   * Return values at all quadrature points                                    \
   */                                                                          \
  template <typename ResultScalarType>                                         \
  return_type<ResultScalarType> operator()(                                    \
    const FEValuesBase<dim, spacedim> &fe_values) const                        \
  {                                                                            \
    return_type<ResultScalarType> out;                                         \
    out.reserve(fe_values.n_quadrature_points);                                \
                                                                               \
    for (const auto &q_point : fe_values.get_quadrature_points())              \
      out.emplace_back(this->template operator()<ResultScalarType>(q_point));  \
                                                                               \
    return out;                                                                \
  }                                                                            \
                                                                               \
  template <typename ResultScalarType>                                         \
  return_type<ResultScalarType> operator()(                                    \
    const FEInterfaceValues<dim, spacedim> &fe_interface_values) const         \
  {                                                                            \
    return_type<ResultScalarType> out;                                         \
    out.reserve(fe_interface_values.n_quadrature_points);                      \
                                                                               \
    for (const auto &q_point : fe_interface_values.get_quadrature_points())    \
      out.emplace_back(this->template operator()<ResultScalarType>(q_point));  \
                                                                               \
    return out;                                                                \
  }                                                                            \
                                                                               \
  /**                                                                          \
   * Return a vectorized set of values for a given quadrature point range.     \
   */                                                                          \
  template <typename ResultScalarType, std::size_t width>                      \
  vectorized_return_type<ResultScalarType, width> operator()(                  \
    const FEValuesBase<dim, spacedim> & fe_values,                             \
    const types::vectorized_qp_range_t &q_point_range) const                   \
  {                                                                            \
    vectorized_return_type<ResultScalarType, width> out;                       \
    Assert(q_point_range.size() <= width,                                      \
           ExcIndexRange(q_point_range.size(), 0, width));                     \
                                                                               \
    /* TODO: Can we guarantee that the underlying function is immutable?  */   \
    DEAL_II_OPENMP_SIMD_PRAGMA                                                 \
    for (unsigned int i = 0; i < q_point_range.size(); ++i)                    \
      if (q_point_range[i] < fe_values.n_quadrature_points)                    \
        numbers::set_vectorized_values(                                        \
          out,                                                                 \
          i,                                                                   \
          this->template operator()<ResultScalarType>(                         \
            fe_values.quadrature_point(q_point_range[i])));                    \
                                                                               \
    return out;                                                                \
  }                                                                            \
                                                                               \
  /**                                                                          \
   * Return a vectorized set of values for a given quadrature point range.     \
   */                                                                          \
  template <typename ResultScalarType, std::size_t width>                      \
  vectorized_return_type<ResultScalarType, width> operator()(                  \
    const FEInterfaceValues<dim, spacedim> &fe_interface_values,               \
    const types::vectorized_qp_range_t &    q_point_range) const                   \
  {                                                                            \
    vectorized_return_type<ResultScalarType, width> out;                       \
    Assert(q_point_range.size() <= width,                                      \
           ExcIndexRange(q_point_range.size(), 0, width));                     \
                                                                               \
    /* TODO: Can we guarantee that the underlying function is immutable?  */   \
    DEAL_II_OPENMP_SIMD_PRAGMA                                                 \
    for (unsigned int i = 0; i < q_point_range.size(); ++i)                    \
      if (q_point_range[i] < fe_interface_values.n_quadrature_points)          \
        numbers::set_vectorized_values(                                        \
          out,                                                                 \
          i,                                                                   \
          this->template operator()<ResultScalarType>(                         \
            fe_interface_values.quadrature_point(q_point_range[i])));          \
                                                                               \
    return out;                                                                \
  }                                                                            \
                                                                               \
  DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_FUNCTION_FUNCTOR_IMPL()               \
                                                                               \
private:                                                                       \
  const Op                                            operand;                 \
  const SmartPointer<const function_type<ScalarType>> function;                \
  const UpdateFlags                                   update_flags;


#define DEAL_II_SYMBOLIC_OP_FUNCTION_FUNCTOR_VALUE_COMMON_IMPL()               \
public:                                                                        \
  template <typename ResultScalarType>                                         \
  using value_type = typename Op::template value_type<ResultScalarType>;       \
                                                                               \
  std::string as_ascii(const SymbolicDecorations &decorator) const             \
  {                                                                            \
    const auto &naming = decorator.get_naming_ascii().differential_operators;  \
    return decorator.decorate_with_operator_ascii(                             \
      naming.value, operand.as_ascii(decorator));                              \
  }                                                                            \
                                                                               \
  std::string as_latex(const SymbolicDecorations &decorator) const             \
  {                                                                            \
    const auto &naming = decorator.get_naming_latex().differential_operators;  \
    return decorator.decorate_with_operator_latex(                             \
      naming.value, operand.as_latex(decorator));                              \
  }                                                                            \
                                                                               \
  DEAL_II_SYMBOLIC_OP_FUNCTION_FUNCTOR_COMMON_IMPL()                           \
                                                                               \
private:                                                                       \
  /**                                                                          \
   * Return single entry                                                       \
   */                                                                          \
  template <typename ResultScalarType>                                         \
  value_type<ResultScalarType> operator()(const Point<spacedim> &point) const  \
  {                                                                            \
    /* We require this try-catch block because the user may not                \
     * overload the value function, but rather chooses to implement            \
     * the value_list function directly.                                       \
     */                                                                        \
    /*                                                                         \
   try                                                                         \
     {                                                                         \
       const value_type<ResultScalarType> value = function->value(point);      \
       return value;                                                           \
     }                                                                         \
   catch (...)                                                                 \
   */                                                                          \
    {                                                                          \
      /* The ResultScalarType might not be compatible with the RangeNumberType \
       * of the function.                                                      \
       */                                                                      \
      /* using Result_t = value_type<ResultScalarType>; */                     \
      using Result_t = decltype(function->value(point));                       \
      std::vector<Result_t> values(1);                                         \
      function->value_list({point}, values);                                   \
      return values[0];                                                        \
    }                                                                          \
  }

#define DEAL_II_SYMBOLIC_OP_FUNCTION_FUNCTOR_GRADIENT_COMMON_IMPL()            \
public:                                                                        \
  template <typename ResultScalarType>                                         \
  using value_type = typename Op::template gradient_type<ResultScalarType>;    \
                                                                               \
  std::string as_ascii(const SymbolicDecorations &decorator) const             \
  {                                                                            \
    const auto &naming = decorator.get_naming_ascii().differential_operators;  \
    return decorator.decorate_with_operator_ascii(                             \
      naming.gradient, operand.as_ascii(decorator));                           \
  }                                                                            \
                                                                               \
  std::string as_latex(const SymbolicDecorations &decorator) const             \
  {                                                                            \
    const auto &naming = decorator.get_naming_latex().differential_operators;  \
    return decorator.decorate_with_operator_latex(                             \
      naming.gradient, operand.as_latex(decorator));                           \
  }                                                                            \
                                                                               \
  DEAL_II_SYMBOLIC_OP_FUNCTION_FUNCTOR_COMMON_IMPL()                           \
                                                                               \
private:                                                                       \
  /**                                                                          \
   * Return single entry                                                       \
   */                                                                          \
  template <typename ResultScalarType>                                         \
  value_type<ResultScalarType> operator()(const Point<spacedim> &point) const  \
  {                                                                            \
    /* We require this try-catch block because the user may not                \
     * overload the gradient function, but rather chooses to implement         \
     * the gradient_list function directly.                                    \
     */                                                                        \
    /*                                                                         \
   try                                                                         \
     {                                                                         \
       const value_type<ResultScalarType> gradient =                           \
         function->gradient(point);                                            \
       return gradient;                                                        \
     }                                                                         \
   catch (...)                                                                 \
   */                                                                          \
    {                                                                          \
      /* The ResultScalarType might not be compatible with the RangeNumberType \
       * of the function.                                                      \
       */                                                                      \
      /* using Result_t = value_type<ResultScalarType>; */                     \
      using Result_t = decltype(function->gradient(point));                    \
      std::vector<Result_t> gradients(1);                                      \
      function->gradient_list({point}, gradients);                             \
      return gradients[0];                                                     \
    }                                                                          \
  }


    /**
     * Extract the value from a scalar function functor.
     *
     * @note This class stores a reference to the function that will be evaluated.
     */
    template <typename ScalarType, int dim, int spacedim>
    class SymbolicOp<ScalarFunctionFunctor<spacedim>,
                     SymbolicOpCodes::value,
                     ScalarType,
                     WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = ScalarFunctionFunctor<spacedim>;
      DEAL_II_SYMBOLIC_OP_FUNCTION_FUNCTOR_VALUE_COMMON_IMPL()

    public:
      static const int rank = 0;
    };


    /**
     * Extract the gradient from a scalar function functor.
     *
     * @note This class stores a reference to the function that will be evaluated.
     */
    template <typename ScalarType, int dim, int spacedim>
    class SymbolicOp<ScalarFunctionFunctor<spacedim>,
                     SymbolicOpCodes::gradient,
                     ScalarType,
                     WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = ScalarFunctionFunctor<spacedim>;
      DEAL_II_SYMBOLIC_OP_FUNCTION_FUNCTOR_GRADIENT_COMMON_IMPL()

    public:
      static const int rank = 1;

      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");
    };



    /**
     * Extract the value from a tensor function functor.
     *
     * @note This class stores a reference to the function that will be evaluated.
     */
    template <typename ScalarType, int rank_, int dim, int spacedim>
    class SymbolicOp<TensorFunctionFunctor<rank_, spacedim>,
                     SymbolicOpCodes::value,
                     ScalarType,
                     WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = TensorFunctionFunctor<rank_, spacedim>;
      DEAL_II_SYMBOLIC_OP_FUNCTION_FUNCTOR_VALUE_COMMON_IMPL()

    public:
      static const int rank = rank_;

      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");
    };



    /**
     * Extract the gradient from a tensor function functor.
     *
     * @note This class stores a reference to the function that will be evaluated.
     */
    template <typename ScalarType, int rank_, int dim, int spacedim>
    class SymbolicOp<TensorFunctionFunctor<rank_, spacedim>,
                     SymbolicOpCodes::gradient,
                     ScalarType,
                     WeakForms::internal::DimPack<dim, spacedim>>
    {
      using Op = TensorFunctionFunctor<rank_, spacedim>;
      DEAL_II_SYMBOLIC_OP_FUNCTION_FUNCTOR_GRADIENT_COMMON_IMPL()

    public:
      static const int rank = rank_ + 1;

      static_assert(value_type<double>::rank == rank,
                    "Mismatch in rank of return value type.");
    };

#undef DEAL_II_SYMBOLIC_OP_FUNCTION_FUNCTOR_GRADIENT_COMMON_IMPL
#undef DEAL_II_SYMBOLIC_OP_FUNCTION_FUNCTOR_VALUE_COMMON_IMPL
#undef DEAL_II_SYMBOLIC_OP_FUNCTION_FUNCTOR_COMMON_IMPL

#undef DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_FUNCTION_FUNCTOR_IMPL
#undef DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_FUNCTOR_IMPL
#undef DEAL_II_SYMBOLIC_EXPRESSION_CONVERSION_COMMON_IMPL

  } // namespace Operators
} // namespace WeakForms



/* ==================== Class method definitions ==================== */



namespace WeakForms
{
  template <typename ScalarType, int dim, int spacedim>
  DEAL_II_ALWAYS_INLINE inline auto
  WeakForms::ScalarFunctor::value(
    const typename WeakForms::ScalarFunctor::
      template function_type<ScalarType, dim, spacedim> &function,
    const typename WeakForms::ScalarFunctor::
      template interface_function_type<ScalarType, dim, spacedim>
        &             interface_function,
    const UpdateFlags update_flags) const
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = ScalarFunctor;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::value,
                              ScalarType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    const auto &operand = *this;
    return OpType(operand, function, interface_function, update_flags);
  }



  template <int rank, int spacedim>
  template <typename ScalarType, int dim>
  DEAL_II_ALWAYS_INLINE inline auto
  WeakForms::TensorFunctor<rank, spacedim>::value(
    const typename WeakForms::TensorFunctor<rank, spacedim>::
      template function_type<ScalarType, dim> &function,
    const typename WeakForms::TensorFunctor<rank, spacedim>::
      template interface_function_type<ScalarType, dim> &interface_function,
    const UpdateFlags                                    update_flags) const
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TensorFunctor<rank, spacedim>;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::value,
                              ScalarType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    const auto &operand = *this;
    return OpType(operand, function, interface_function, update_flags);
  }



  template <int rank, int spacedim>
  template <typename ScalarType, int dim>
  DEAL_II_ALWAYS_INLINE inline auto
  WeakForms::SymmetricTensorFunctor<rank, spacedim>::value(
    const typename WeakForms::SymmetricTensorFunctor<rank, spacedim>::
      template function_type<ScalarType, dim> &function,
    const typename WeakForms::SymmetricTensorFunctor<rank, spacedim>::
      template interface_function_type<ScalarType, dim> &interface_function,
    const UpdateFlags                                    update_flags) const
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = SymmetricTensorFunctor<rank, spacedim>;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::value,
                              ScalarType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    const auto &operand = *this;
    return OpType(operand, function, interface_function, update_flags);
  }



  template <int spacedim>
  template <typename ScalarType, int dim>
  DEAL_II_ALWAYS_INLINE inline auto
  WeakForms::ScalarFunctionFunctor<spacedim>::value(
    const typename WeakForms::ScalarFunctionFunctor<
      spacedim>::template function_type<ScalarType> &function,
    const UpdateFlags                                update_flags) const
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = ScalarFunctionFunctor<spacedim>;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::value,
                              ScalarType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    const auto &operand = *this;
    return OpType(operand, function, update_flags);
  }



  template <int spacedim>
  template <typename ScalarType, int dim>
  DEAL_II_ALWAYS_INLINE inline auto
  WeakForms::ScalarFunctionFunctor<spacedim>::gradient(
    const typename WeakForms::ScalarFunctionFunctor<
      spacedim>::template function_type<ScalarType> &function,
    const UpdateFlags                                update_flags) const
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = ScalarFunctionFunctor<spacedim>;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::gradient,
                              ScalarType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    const auto &operand = *this;
    return OpType(operand, function, update_flags);
  }



  template <int rank, int spacedim>
  template <typename ScalarType, int dim>
  DEAL_II_ALWAYS_INLINE inline auto
  WeakForms::TensorFunctionFunctor<rank, spacedim>::value(
    const typename WeakForms::TensorFunctionFunctor<rank, spacedim>::
      template function_type<ScalarType> &function,
    const UpdateFlags                     update_flags) const
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TensorFunctionFunctor<rank, spacedim>;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::value,
                              ScalarType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    const auto &operand = *this;
    return OpType(operand, function, update_flags);
  }



  template <int rank, int spacedim>
  template <typename ScalarType, int dim>
  DEAL_II_ALWAYS_INLINE inline auto
  WeakForms::TensorFunctionFunctor<rank, spacedim>::gradient(
    const typename WeakForms::TensorFunctionFunctor<rank, spacedim>::
      template function_type<ScalarType> &function,
    const UpdateFlags                     update_flags) const
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op     = TensorFunctionFunctor<rank, spacedim>;
    using OpType = SymbolicOp<Op,
                              SymbolicOpCodes::gradient,
                              ScalarType,
                              WeakForms::internal::DimPack<dim, spacedim>>;

    const auto &operand = *this;
    return OpType(operand, function, update_flags);
  }
} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  /**
   * @brief A convenience function that always returns the same constant scalar
   * value upon evaluation.
   *
   * This variation allows one to prescribe the value, as well as the ASCII and
   * LaTeX representation of the value.
   *
   * An example of usage:
   * @code {.cpp}
   * const auto i =
   *   constant_scalar<dim>(std::complex<double>{0, 1}, "i", "i");
   * @endcode
   *
   * @tparam dim The dimension in which the scalar is being evaluated.
   * @tparam spacedim The spatial dimension in which the scalar is being evaluated.
   * @tparam ScalarType The underlying scalar type.
   * @param value The value to be returned upon evaluation.
   * @param symbol_ascii The ASCII representation of the value.
   * @param symbol_latex  The LaTeX representation of the value.
   * @return auto Returns a symbolic operator based on a ScalarFunctor.
   *
   * \ingroup functors convenience_functions
   */
  template <int dim, int spacedim = dim, typename ScalarType>
  auto
  constant_scalar(const ScalarType & value,
                  const std::string &symbol_ascii,
                  const std::string &symbol_latex)
  {
    const ScalarFunctor functor(symbol_ascii, symbol_latex);

    return functor.template value<ScalarType, dim, spacedim>(
      [value](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return value; },
      [value](const FEInterfaceValues<dim, spacedim> &, const unsigned int)
      { return value; });
  }



  /**
   * @brief A convenience function that always returns the same constant scalar
   * value upon evaluation.
   *
   * This variation allows one to prescribe only the value; the ASCII and LaTeX
   * representations exactly match the value itself.
   *
   * An example of usage:
   * @code {.cpp}
   * const auto s = constant_scalar<dim>(1.0);
   * @endcode
   *
   * @tparam dim The dimension in which the scalar is being evaluated.
   * @tparam spacedim The spatial dimension in which the scalar is being evaluated.
   * @tparam ScalarType The underlying scalar type.
   * @param value The value to be returned upon evaluation.
   * @return auto Returns a symbolic operator based on a ScalarFunctor.
   *
   * \ingroup functors convenience_functions
   */
  template <int dim, int spacedim = dim, typename ScalarType>
  auto
  constant_scalar(const ScalarType &value)
  {
    using Converter = WeakForms::Utilities::ConvertNumericToText<ScalarType>;
    return constant_scalar<dim, spacedim>(value,
                                          Converter::to_ascii(value),
                                          Converter::to_latex(value));
  }



  /**
   * @brief A convenience function that always returns the same constant tensor
   * value upon evaluation.
   *
   * This variation allows one to prescribe the value, as well as the ASCII and
   * LaTeX representation of the tensor.
   *
   * An example of usage:
   * @code {.cpp}
   * const auto T = constant_tensor<dim>(Tensor<2, dim>{}, "T", "T");
   * @endcode
   *
   * @tparam dim The dimension in which the tensor is being evaluated.
   * @tparam rank The rank of the tensor that is returned upon evaluation.
   * @tparam spacedim The spatial dimension in which the tensor is being evaluated.
   * @tparam ScalarType The underlying scalar type for each component of the tensor.
   * @param value The value to be returned upon evaluation.
   * @param symbol_ascii The ASCII representation of the tensor.
   * @param symbol_latex  The LaTeX representation of the tensor.
   * @return auto Returns a symbolic operator based on a TensorFunctor.
   *
   * \ingroup functors convenience_functions
   */
  template <int dim, int rank, int spacedim, typename ScalarType>
  auto
  constant_tensor(const Tensor<rank, spacedim, ScalarType> &value,
                  const std::string &                       symbol_ascii,
                  const std::string &                       symbol_latex)
  {
    const TensorFunctor<rank, spacedim> functor(symbol_ascii, symbol_latex);

    return functor.template value<ScalarType, dim>(
      [value](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return value; },
      [value](const FEInterfaceValues<dim, spacedim> &, const unsigned int)
      { return value; });
  }



  /**
   * @brief A convenience function that always returns the same constant tensor
   * value upon evaluation.
   *
   * This variation allows one to prescribe only the values of the tensor
   * components; the ASCII and LaTeX representations are a flattened
   * representation of the tensor.
   *
   * An example of usage:
   * @code {.cpp}
   * const auto T = constant_tensor<dim>(Tensor<2, dim>{});
   * @endcode
   *
   * @tparam dim The dimension in which the tensor is being evaluated.
   * @tparam rank The rank of the tensor that is returned upon evaluation.
   * @tparam spacedim The spatial dimension in which the tensor is being evaluated.
   * @tparam ScalarType The underlying scalar type for each component of the tensor.
   * @param value The value to be returned upon evaluation.
   * @return auto Returns a symbolic operator based on a TensorFunctor.
   *
   * \ingroup functors convenience_functions
   */
  template <int dim, int rank, int spacedim, typename ScalarType>
  auto
  constant_tensor(const Tensor<rank, spacedim, ScalarType> &value)
  {
    using Converter = WeakForms::Utilities::ConvertNumericToText<
      Tensor<rank, spacedim, ScalarType>>;
    return constant_tensor<dim>(value,
                                Converter::to_ascii(value),
                                Converter::to_latex(value));
  }



  /**
   * @brief A convenience function that always returns the same constant vector
   * (rank-1 tensor) value upon evaluation.
   *
   * This variation allows one to prescribe the value, as well as the ASCII and
   * LaTeX representation of the vector.
   *
   * An example of usage:
   * @code {.cpp}
   * const auto v = constant_vector<dim>(Tensor<1, dim>(), "v", "v");
   * @endcode
   *
   * @tparam dim The dimension in which the tensor is being evaluated.
   * @tparam spacedim The spatial dimension in which the tensor is being evaluated.
   * @tparam ScalarType The underlying scalar type for each component of the tensor.
   * @param value The value to be returned upon evaluation.
   * @param symbol_ascii The ASCII representation of the vector.
   * @param symbol_latex  The LaTeX representation of the vector.
   * @return auto Returns a symbolic operator based on a rank-1 TensorFunctor.
   *
   * \ingroup functors convenience_functions
   */
  template <int dim, int spacedim, typename ScalarType>
  auto
  constant_vector(const Tensor<1, spacedim, ScalarType> &value,
                  const std::string &                    symbol_ascii,
                  const std::string &                    symbol_latex)
  {
    return constant_tensor<dim>(value, symbol_ascii, symbol_latex);
  }



  /**
   * @brief A convenience function that always returns the same constant vector
   * (rank-1 tensor) value upon evaluation.
   *
   * This variation allows one to prescribe only the values of the vector
   * components; the ASCII and LaTeX representations are a flattened
   * representation of the vector.
   *
   * An example of usage:
   * @code {.cpp}
   * const auto v = constant_vector<dim>(Tensor<1, dim>());
   * @endcode
   *
   * @tparam dim The dimension in which the tensor is being evaluated.
   * @tparam spacedim The spatial dimension in which the tensor is being evaluated.
   * @tparam ScalarType The underlying scalar type for each component of the tensor.
   * @param value The value to be returned upon evaluation.
   * @return auto Returns a symbolic operator based on a rank-1 TensorFunctor.
   *
   * \ingroup functors convenience_functions
   */
  template <int dim, int spacedim, typename ScalarType>
  auto
  constant_vector(const Tensor<1, spacedim, ScalarType> &value)
  {
    return constant_tensor<dim>(value);
  }



  /**
   * @brief A convenience function that always returns the same constant symmetric tensor
   * value upon evaluation.
   *
   * This variation allows one to prescribe the value, as well as the ASCII and
   * LaTeX representation of the symmetric tensor.
   *
   * An example of usage:
   * @code {.cpp}
   * const auto S2 = constant_symmetric_tensor<dim>(
   *   unit_symmetric_tensor<dim>(), "S", "S");
   * @endcode
   *
   * @tparam dim The dimension in which the tensor is being evaluated.
   * @tparam rank The rank of the symmetric tensor that is returned upon evaluation.
   * @tparam spacedim The spatial dimension in which the tensor is being evaluated.
   * @tparam ScalarType The underlying scalar type for each component of the tensor.
   * @param value The value to be returned upon evaluation.
   * @param symbol_ascii The ASCII representation of the tensor.
   * @param symbol_latex  The LaTeX representation of the tensor.
   * @return auto Returns a symbolic operator based on a SymmetricTensorFunctor.
   *
   * \ingroup functors convenience_functions
   */
  template <int dim, int rank, int spacedim, typename ScalarType>
  auto
  constant_symmetric_tensor(
    const SymmetricTensor<rank, spacedim, ScalarType> &value,
    const std::string &                                symbol_ascii,
    const std::string &                                symbol_latex)
  {
    const SymmetricTensorFunctor<rank, spacedim> functor(symbol_ascii,
                                                         symbol_latex);

    return functor.template value<ScalarType, dim>(
      [value](const FEValuesBase<dim, spacedim> &, const unsigned int)
      { return value; },
      [value](const FEInterfaceValues<dim, spacedim> &, const unsigned int)
      { return value; });
  }



  /**
   * @brief A convenience function that always returns the same constant symmetric tensor
   * value upon evaluation.
   *
   * This variation allows one to prescribe only the values of the symmetric
   * tensor components; the ASCII and LaTeX representations are a flattened
   * representation of the symmetric tensor.
   *
   * An example of usage:
   * @code {.cpp}
   * const auto S4 = constant_symmetric_tensor<dim>(identity_tensor<dim>());
   * @endcode
   *
   * @tparam dim The dimension in which the tensor is being evaluated.
   * @tparam rank The rank of the symmetric tensor that is returned upon evaluation.
   * @tparam spacedim The spatial dimension in which the tensor is being evaluated.
   * @tparam ScalarType The underlying scalar type for each component of the tensor.
   * @param value The value to be returned upon evaluation.
   * @return auto Returns a symbolic operator based on a SymmetricTensorFunctor.
   *
   * \ingroup functors convenience_functions
   */
  template <int dim, int rank, int spacedim, typename ScalarType>
  auto
  constant_symmetric_tensor(
    const SymmetricTensor<rank, spacedim, ScalarType> &value)
  {
    using Converter = WeakForms::Utilities::ConvertNumericToText<
      SymmetricTensor<rank, spacedim, ScalarType>>;
    return constant_symmetric_tensor<dim>(value,
                                          Converter::to_ascii(value),
                                          Converter::to_latex(value));
  }
} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  // Unary operations
  template <typename ScalarType, int dim, int spacedim>
  struct is_functor_op<WeakForms::Operators::SymbolicOp<
    WeakForms::ScalarFunctor,
    WeakForms::Operators::SymbolicOpCodes::value,
    ScalarType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};

  template <typename ScalarType, int dim, int rank, int spacedim>
  struct is_functor_op<WeakForms::Operators::SymbolicOp<
    WeakForms::TensorFunctor<rank, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::value,
    ScalarType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};

  template <typename ScalarType, int dim, int rank, int spacedim>
  struct is_functor_op<WeakForms::Operators::SymbolicOp<
    WeakForms::SymmetricTensorFunctor<rank, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::value,
    ScalarType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};

  template <typename ScalarType, int dim, int spacedim>
  struct is_functor_op<WeakForms::Operators::SymbolicOp<
    WeakForms::ScalarFunctionFunctor<spacedim>,
    WeakForms::Operators::SymbolicOpCodes::value,
    ScalarType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};

  template <typename ScalarType, int dim, int spacedim>
  struct is_functor_op<WeakForms::Operators::SymbolicOp<
    WeakForms::ScalarFunctionFunctor<spacedim>,
    WeakForms::Operators::SymbolicOpCodes::gradient,
    ScalarType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};

  template <typename ScalarType, int dim, int rank, int spacedim>
  struct is_functor_op<WeakForms::Operators::SymbolicOp<
    WeakForms::TensorFunctionFunctor<rank, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::value,
    ScalarType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};

  template <typename ScalarType, int dim, int rank, int spacedim>
  struct is_functor_op<WeakForms::Operators::SymbolicOp<
    WeakForms::TensorFunctionFunctor<rank, spacedim>,
    WeakForms::Operators::SymbolicOpCodes::gradient,
    ScalarType,
    internal::DimPack<dim, spacedim>>> : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN



WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_functors_h
