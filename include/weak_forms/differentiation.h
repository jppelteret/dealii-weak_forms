#ifndef dealii_weakforms_differentiation_h
#define dealii_weakforms_differentiation_h

#include <deal.II/base/config.h>

#include <deal.II/base/numbers.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/tensor.h>

#include <weak_forms/cache_functors.h>
#include <weak_forms/config.h>

#include <tuple>
#include <type_traits>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  namespace internal
  {
    // TODO: This is replicated in energy_functor.h
    template <typename T>
    class is_scalar_type
    {
      // See has_begin_and_end() in template_constraints.h
      // and https://stackoverflow.com/a/10722840

      template <typename A>
      static constexpr auto
      test(int) -> decltype(std::declval<typename EnableIfScalar<A>::type>(),
                            std::true_type())
      {
        return true;
      }

      template <typename A>
      static std::false_type
      test(...);

    public:
      using type = decltype(test<T>(0));

      static const bool value = type::value;
    };


    template <typename T, typename U, typename = void>
    struct are_scalar_types : std::false_type
    {};


    template <typename T, typename U>
    struct are_scalar_types<
      T,
      U,
      typename std::enable_if<is_scalar_type<T>::value &&
                              is_scalar_type<U>::value>::type> : std::true_type
    {};


    // Determine types resulting from differential operations
    // of scalars, tensors and symmetric tensors.
    namespace Differentiation
    {
      // A wrapper for functors that helps produce the correct differential
      // notation in the output. Without this, the @p SymbolicDiffOp would have
      // full authority of what is being printed, which means that we have no
      // way of isolating the boldening (when necessary) the functor that's
      // being differentiated and the fields with which it's being
      // differentiated with respect to. All of the differential operation
      // notation, and the functor and fields would carry the notation
      // of the underlying symbolic operator for a
      // [Scalar/Tensor/SymmetricTensor]CacheFunctor.
      //
      // The template parameter @p FunctorOp is the functor that has been
      // differentiated, and the @p FieldOps are the field operators that its
      // being differentiated with respect to.
      template <typename SymbolicFunctorOp,
                typename FunctorOp,
                typename... FieldOps>
      class SymbolicDiffOp : public SymbolicFunctorOp
      {
      public:
        SymbolicDiffOp(const SymbolicFunctorOp &symbolic_functor_op,
                       const FunctorOp &        functor_op,
                       const FieldOps &... field_ops)
          : SymbolicFunctorOp(symbolic_functor_op)
          , functor_op(functor_op)
          , field_ops(field_ops...)
        {}

        std::string
        as_ascii(const SymbolicDecorations &decorator) const
        {
          const auto &naming =
            decorator.get_naming_ascii().differential_operators;
          return decorator.decorate_with_operator_ascii(
            naming.value,
            decorator.symbolic_op_derivative_as_ascii(functor_op, field_ops));
        }

        std::string
        as_latex(const SymbolicDecorations &decorator) const
        {
          const auto &naming =
            decorator.get_naming_latex().differential_operators;
          return decorator.decorate_with_operator_latex(
            naming.value,
            decorator.symbolic_op_derivative_as_latex(functor_op, field_ops));
        }

        // Expose base class definitions
        using SymbolicFunctorOp::get_update_flags;
        using SymbolicFunctorOp::operator();

      private:
        const FunctorOp               functor_op;
        const std::tuple<FieldOps...> field_ops;
      };


      template <typename T, typename U, typename = void>
      struct DiffOpResult;

      // Differentiate a scalar with respect to another scalar
      template <typename T, typename U>
      struct DiffOpResult<
        T,
        U,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = 0;
        using scalar_type         = typename ProductType<T, U>::type;
        using type                = scalar_type;

        using Op = WeakForms::ScalarCacheFunctor;
        template <int dim, int spacedim = dim>
        using function_type =
          typename Op::template function_type<scalar_type, dim, spacedim>;

        // Return symbolic op to functor
        template <int dim,
                  int spacedim,
                  typename FunctorOp,
                  typename... FieldOps>
        static auto
        get_functor(const std::string &                 symbol_ascii,
                    const std::string &                 symbol_latex,
                    const function_type<dim, spacedim> &function,
                    const FunctorOp &                   functor_op,
                    const FieldOps &... field_ops)
        {
          const auto symbolic_op =
            get_operand(symbol_ascii, symbol_latex)
              .template value<scalar_type, dim, spacedim>(
                function, UpdateFlags::update_default);
          using SymbolicOp_t = typename std::decay<decltype(symbolic_op)>::type;
          return SymbolicDiffOp<SymbolicOp_t, FunctorOp, FieldOps...>(
            symbolic_op, functor_op, field_ops...);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Differentiate a scalar with respect to a tensor
      template <int rank_, int spacedim, typename T, typename U>
      struct DiffOpResult<
        T,
        Tensor<rank_, spacedim, U>,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = rank_;
        using scalar_type         = typename ProductType<T, U>::type;
        using type                = Tensor<rank, spacedim, scalar_type>;

        using Op = TensorCacheFunctor<rank, spacedim>;
        template <int dim = spacedim>
        using function_type =
          typename Op::template function_type<scalar_type, dim>;

        // Return symbolic op to functor
        template <int dim,
                  int /*spacedim*/,
                  typename FunctorOp,
                  typename... FieldOps>
        static auto
        get_functor(const std::string &       symbol_ascii,
                    const std::string &       symbol_latex,
                    const function_type<dim> &function,
                    const FunctorOp &         functor_op,
                    const FieldOps &... field_ops)
        {
          const auto symbolic_op =
            get_operand(symbol_ascii, symbol_latex)
              .template value<scalar_type, dim>(function,
                                                UpdateFlags::update_default);
          using SymbolicOp_t = typename std::decay<decltype(symbolic_op)>::type;
          return SymbolicDiffOp<SymbolicOp_t, FunctorOp, FieldOps...>(
            symbolic_op, functor_op, field_ops...);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Differentiate a scalar with respect to a symmetric tensor
      template <int rank_, int spacedim, typename T, typename U>
      struct DiffOpResult<
        T,
        SymmetricTensor<rank_, spacedim, U>,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = rank_;
        using scalar_type         = typename ProductType<T, U>::type;
        using type = SymmetricTensor<rank, spacedim, scalar_type>;

        using Op = SymmetricTensorCacheFunctor<rank, spacedim>;
        template <int dim = spacedim>
        using function_type =
          typename Op::template function_type<scalar_type, dim>;

        // Return symbolic op to functor
        template <int dim,
                  int /*spacedim*/,
                  typename FunctorOp,
                  typename... FieldOps>
        static auto
        get_functor(const std::string &       symbol_ascii,
                    const std::string &       symbol_latex,
                    const function_type<dim> &function,
                    const FunctorOp &         functor_op,
                    const FieldOps &... field_ops)
        {
          const auto symbolic_op =
            get_operand(symbol_ascii, symbol_latex)
              .template value<scalar_type, dim>(function,
                                                UpdateFlags::update_default);
          using SymbolicOp_t = typename std::decay<decltype(symbolic_op)>::type;
          return SymbolicDiffOp<SymbolicOp_t, FunctorOp, FieldOps...>(
            symbolic_op, functor_op, field_ops...);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Differentiate a tensor with respect to a scalar
      template <int rank_, int spacedim, typename T, typename U>
      struct DiffOpResult<
        Tensor<rank_, spacedim, T>,
        U,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = rank_;
        using scalar_type         = typename ProductType<T, U>::type;
        using type                = Tensor<rank, spacedim, scalar_type>;

        using Op = TensorCacheFunctor<rank, spacedim>;
        template <int dim = spacedim>
        using function_type =
          typename Op::template function_type<scalar_type, dim>;

        // Return symbolic op to functor
        template <int dim,
                  int /*spacedim*/,
                  typename FunctorOp,
                  typename... FieldOps>
        static auto
        get_functor(const std::string &       symbol_ascii,
                    const std::string &       symbol_latex,
                    const function_type<dim> &function,
                    const FunctorOp &         functor_op,
                    const FieldOps &... field_ops)
        {
          const auto symbolic_op =
            get_operand(symbol_ascii, symbol_latex)
              .template value<scalar_type, dim>(function,
                                                UpdateFlags::update_default);
          using SymbolicOp_t = typename std::decay<decltype(symbolic_op)>::type;
          return SymbolicDiffOp<SymbolicOp_t, FunctorOp, FieldOps...>(
            symbolic_op, functor_op, field_ops...);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Differentiate a tensor with respect to another tensor
      template <int rank_1, int rank_2, int spacedim, typename T, typename U>
      struct DiffOpResult<
        Tensor<rank_1, spacedim, T>,
        Tensor<rank_2, spacedim, U>,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = rank_1 + rank_2;
        using scalar_type         = typename ProductType<T, U>::type;
        using type                = Tensor<rank, spacedim, scalar_type>;

        using Op = TensorCacheFunctor<rank, spacedim>;
        template <int dim = spacedim>
        using function_type =
          typename Op::template function_type<scalar_type, dim>;

        // Return symbolic op to functor
        template <int dim,
                  int /*spacedim*/,
                  typename FunctorOp,
                  typename... FieldOps>
        static auto
        get_functor(const std::string &       symbol_ascii,
                    const std::string &       symbol_latex,
                    const function_type<dim> &function,
                    const FunctorOp &         functor_op,
                    const FieldOps &... field_ops)
        {
          const auto symbolic_op =
            get_operand(symbol_ascii, symbol_latex)
              .template value<scalar_type, dim>(function,
                                                UpdateFlags::update_default);
          using SymbolicOp_t = typename std::decay<decltype(symbolic_op)>::type;
          return SymbolicDiffOp<SymbolicOp_t, FunctorOp, FieldOps...>(
            symbolic_op, functor_op, field_ops...);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Differentiate a tensor with respect to a symmetric tensor
      template <int rank_1, int rank_2, int spacedim, typename T, typename U>
      struct DiffOpResult<
        Tensor<rank_1, spacedim, T>,
        SymmetricTensor<rank_2, spacedim, U>,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = rank_1 + rank_2;
        using scalar_type         = typename ProductType<T, U>::type;
        using type                = Tensor<rank, spacedim, scalar_type>;

        using Op = TensorCacheFunctor<rank, spacedim>;
        template <int dim = spacedim>
        using function_type =
          typename Op::template function_type<scalar_type, dim>;

        // Return symbolic op to functor
        template <int dim,
                  int /*spacedim*/,
                  typename FunctorOp,
                  typename... FieldOps>
        static auto
        get_functor(const std::string &       symbol_ascii,
                    const std::string &       symbol_latex,
                    const function_type<dim> &function,
                    const FunctorOp &         functor_op,
                    const FieldOps &... field_ops)
        {
          const auto symbolic_op =
            get_operand(symbol_ascii, symbol_latex)
              .template value<scalar_type, dim>(function,
                                                UpdateFlags::update_default);
          using SymbolicOp_t = typename std::decay<decltype(symbolic_op)>::type;
          return SymbolicDiffOp<SymbolicOp_t, FunctorOp, FieldOps...>(
            symbolic_op, functor_op, field_ops...);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Differentiate a symmetric tensor with respect to a scalar
      template <int rank_, int spacedim, typename T, typename U>
      struct DiffOpResult<
        SymmetricTensor<rank_, spacedim, T>,
        U,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = rank_;
        using scalar_type         = typename ProductType<T, U>::type;
        using type = SymmetricTensor<rank_, spacedim, scalar_type>;

        using Op = SymmetricTensorCacheFunctor<rank, spacedim>;
        template <int dim = spacedim>
        using function_type =
          typename Op::template function_type<scalar_type, dim>;

        // Return symbolic op to functor
        template <int dim,
                  int /*spacedim*/,
                  typename FunctorOp,
                  typename... FieldOps>
        static auto
        get_functor(const std::string &       symbol_ascii,
                    const std::string &       symbol_latex,
                    const function_type<dim> &function,
                    const FunctorOp &         functor_op,
                    const FieldOps &... field_ops)
        {
          const auto symbolic_op =
            get_operand(symbol_ascii, symbol_latex)
              .template value<scalar_type, dim>(function,
                                                UpdateFlags::update_default);
          using SymbolicOp_t = typename std::decay<decltype(symbolic_op)>::type;
          return SymbolicDiffOp<SymbolicOp_t, FunctorOp, FieldOps...>(
            symbolic_op, functor_op, field_ops...);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Differentiate a symmetric tensor with respect to a tensor
      template <int rank_1, int rank_2, int spacedim, typename T, typename U>
      struct DiffOpResult<
        SymmetricTensor<rank_1, spacedim, T>,
        Tensor<rank_2, spacedim, U>,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = rank_1 + rank_2;
        using scalar_type         = typename ProductType<T, U>::type;
        using type                = Tensor<rank, spacedim, scalar_type>;

        using Op = TensorCacheFunctor<rank, spacedim>;
        template <int dim = spacedim>
        using function_type =
          typename Op::template function_type<scalar_type, dim>;

        // Return symbolic op to functor
        template <int dim,
                  int /*spacedim*/,
                  typename FunctorOp,
                  typename... FieldOps>
        static auto
        get_functor(const std::string &       symbol_ascii,
                    const std::string &       symbol_latex,
                    const function_type<dim> &function,
                    const FunctorOp &         functor_op,
                    const FieldOps &... field_ops)
        {
          const auto symbolic_op =
            get_operand(symbol_ascii, symbol_latex)
              .template value<scalar_type, dim>(function,
                                                UpdateFlags::update_default);
          using SymbolicOp_t = typename std::decay<decltype(symbolic_op)>::type;
          return SymbolicDiffOp<SymbolicOp_t, FunctorOp, FieldOps...>(
            symbolic_op, functor_op, field_ops...);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Differentiate a symmetric tensor with respect to another symmetric
      // tensor
      template <int rank_1, int rank_2, int spacedim, typename T, typename U>
      struct DiffOpResult<
        SymmetricTensor<rank_1, spacedim, T>,
        SymmetricTensor<rank_2, spacedim, U>,
        typename std::enable_if<are_scalar_types<T, U>::value>::type>
      {
        static constexpr int rank = rank_1 + rank_2;
        using scalar_type         = typename ProductType<T, U>::type;
        using type = SymmetricTensor<rank, spacedim, scalar_type>;

        using Op = SymmetricTensorCacheFunctor<rank, spacedim>;
        template <int dim = spacedim>
        using function_type =
          typename Op::template function_type<scalar_type, dim>;

        // Return symbolic op to functor
        template <int dim,
                  int /*spacedim*/,
                  typename FunctorOp,
                  typename... FieldOps>
        static auto
        get_functor(const std::string &       symbol_ascii,
                    const std::string &       symbol_latex,
                    const function_type<dim> &function,
                    const FunctorOp &         functor_op,
                    const FieldOps &... field_ops)
        {
          const auto symbolic_op =
            get_operand(symbol_ascii, symbol_latex)
              .template value<scalar_type, dim>(function,
                                                UpdateFlags::update_default);
          using SymbolicOp_t = typename std::decay<decltype(symbolic_op)>::type;
          return SymbolicDiffOp<SymbolicOp_t, FunctorOp, FieldOps...>(
            symbolic_op, functor_op, field_ops...);
        }

      private:
        static Op
        get_operand(const std::string &symbol_ascii,
                    const std::string &symbol_latex)
        {
          return Op(symbol_ascii, symbol_latex);
        }
      };

      // Specialization for differentiating a scalar or tensor with respect
      // to a tuple of fields.
      // This is only intended for use in a very narrow context, so we only
      // define the result types for the operation.
      //
      // This builds up into a tuple that can be represented as:
      // [ df/dx_1 , df/dx_2 , ... , df/dx_n ]
      template <typename T, typename... Us>
      struct DiffOpResult<T, std::tuple<Us...>, void>
      {
        // The result type
        using type = std::tuple<typename DiffOpResult<T, Us>::type...>;
      };

      // Specialization for differentiating a tuple of scalars or tensors
      // with respect to a tuple of fields.
      // This is only intended for use in a very narrow context, so we only
      // define the result types for the operation.
      //
      // This builds up into a nested tuple with the following
      // structure/grouping:
      // [ [ d^2f/dx_1.dx_1 , d^2f/dx_1.dx_2 , ... , d^2f/dx_1.dx_n ] ,
      //   [ d^2f/dx_2.dx_1 , d^2f/dx_2.dx_2 , ... , d^2f/dx_2.dx_n ] ,
      //   [                   ...                                  ] ,
      //   [ d^2f/dx_n.dx_1 , d^2f/dx_n.dx_2 , ... , d^2f/dx_n.dx_n ] ]
      //
      // So the outer tuple holds the "row elements", and the inner tuple
      // the "column elements" for each row.
      template <typename... Ts, typename... Us>
      struct DiffOpResult<std::tuple<Ts...>, std::tuple<Us...>, void>
      {
        // The result type
        using type =
          std::tuple<typename DiffOpResult<Ts, std::tuple<Us...>>::type...>;
      };

    } // namespace Differentiation
  }   // namespace internal
} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  // Unary operations
  template <typename SymbolicFunctorOp,
            typename FunctorOp,
            typename... FieldOps>
  struct is_cache_functor_op<
    internal::Differentiation::
      SymbolicDiffOp<SymbolicFunctorOp, FunctorOp, FieldOps...>>
    : std::true_type
  {};

  template <typename SymbolicFunctorOp,
            typename FunctorOp,
            typename... FieldOps>
  struct is_unary_op<
    internal::Differentiation::
      SymbolicDiffOp<SymbolicFunctorOp, FunctorOp, FieldOps...>>
    : std::true_type
  {};

} // namespace WeakForms


#endif // DOXYGEN


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_differentiation_h
