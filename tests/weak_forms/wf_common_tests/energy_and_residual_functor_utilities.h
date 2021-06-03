
#include <deal.II/base/config.h>

#include <deal.II/base/utilities.h>

#include <deal.II/differentiation/ad.h>
#include <deal.II/differentiation/sd.h>

#include <weak_forms/config.h>
#include <weak_forms/energy_functor.h>
#include <weak_forms/residual_functor.h>
#include <weak_forms/symbolic_decorations.h>


WEAK_FORMS_NAMESPACE_OPEN

namespace WeakForms
{
  namespace Operators
  {
    // ======= AD =======


    template <typename... SymbolicOpsSubSpaceFieldSolution>
    using OpHelper_t = internal::SymbolicOpsSubSpaceFieldSolutionHelper<
      SymbolicOpsSubSpaceFieldSolution...>;

    // End point
    template <std::size_t I = 0, typename... SymbolicOpType>
    static typename std::enable_if<I == sizeof...(SymbolicOpType), void>::type
    unpack_print_ad_functor_field_args_and_extractors(
      const typename OpHelper_t<SymbolicOpType...>::field_args_t &field_args,
      const typename OpHelper_t<SymbolicOpType...>::field_extractors_t
        &                        field_extractors,
      const SymbolicDecorations &decorator)
    {
      (void)field_args;
      (void)field_extractors;
    }


    template <std::size_t I = 0, typename... SymbolicOpType>
      static typename std::enable_if <
      I<sizeof...(SymbolicOpType), void>::type
      unpack_print_ad_functor_field_args_and_extractors(
        const typename OpHelper_t<SymbolicOpType...>::field_args_t &field_args,
        const typename OpHelper_t<SymbolicOpType...>::field_extractors_t
          &                        field_extractors,
        const SymbolicDecorations &decorator)
    {
      deallog << "Field index  " << dealii::Utilities::to_string(I) << ": "
              << std::get<I>(field_args).as_ascii(decorator) << " -> "
              << std::get<I>(field_extractors).get_name() << std::endl;

      unpack_print_ad_functor_field_args_and_extractors<I + 1,
                                                        SymbolicOpType...>(
        field_args, field_extractors, decorator);
    }


    template <typename... SymbolicOpsSubSpaceFieldSolution,
              typename SymbolicOpADFunctor,
              typename = typename std::enable_if<
                WeakForms::is_ad_functor_op<SymbolicOpADFunctor>::value>::type>
    void
    print_ad_functor_field_args_and_extractors_impl(
      const SymbolicOpADFunctor &           ad_functor,
      const WeakForms::SymbolicDecorations &decorator)
    {
      deallog << "Number of components: "
              << dealii::Utilities::to_string(
                   OpHelper_t<
                     SymbolicOpsSubSpaceFieldSolution...>::get_n_components())
              << std::endl;

      unpack_print_ad_functor_field_args_and_extractors<
        0,
        SymbolicOpsSubSpaceFieldSolution...>(ad_functor.get_field_args(),
                                             ad_functor.get_field_extractors(),
                                             decorator);
    }


    template <typename ADNumberType,
              int dim,
              int spacedim,
              typename... SymbolicOpsSubSpaceFieldSolution>
    void
    print_ad_functor_field_args_and_extractors(
      const SymbolicOp<
        WeakForms::EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>,
        SymbolicOpCodes::value,
        typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
        ADNumberType,
        WeakForms::internal::DimPack<dim, spacedim>> &energy_functor,
      const WeakForms::SymbolicDecorations &          decorator)
    {
      print_ad_functor_field_args_and_extractors_impl<
        SymbolicOpsSubSpaceFieldSolution...>(energy_functor, decorator);
    }


    template <typename ADNumberType,
              int dim,
              int spacedim,
              typename TestSpaceOp,
              typename... SymbolicOpsSubSpaceFieldSolution>
    void
    print_ad_functor_field_args_and_extractors(
      const SymbolicOp<
        WeakForms::ResidualViewFunctor<TestSpaceOp,
                                       SymbolicOpsSubSpaceFieldSolution...>,
        SymbolicOpCodes::value,
        typename Differentiation::AD::ADNumberTraits<ADNumberType>::scalar_type,
        ADNumberType,
        WeakForms::internal::DimPack<dim, spacedim>> &residual_functor,
      const WeakForms::SymbolicDecorations &          decorator)
    {
      print_ad_functor_field_args_and_extractors_impl<
        SymbolicOpsSubSpaceFieldSolution...>(residual_functor, decorator);
    }


    // ======= SD =======


    // template <typename... SymbolicOpsSubSpaceFieldSolution>
    // using OpHelper_t = internal::SymbolicOpsSubSpaceFieldSolutionHelper<
    //   SymbolicOpsSubSpaceFieldSolution...>;

    // End point
    template <std::size_t I = 0,
              typename SDNumberType,
              typename... SymbolicOpType>
    static typename std::enable_if<I == sizeof...(SymbolicOpType), void>::type
    unpack_print_sd_functor_print_field_args_and_symbolic_fields(
      const typename OpHelper_t<SymbolicOpType...>::field_args_t &field_args,
      const typename OpHelper_t<SymbolicOpType...>::template field_values_t<
        SDNumberType> &          symbolic_fields,
      const SymbolicDecorations &decorator)
    {
      (void)field_args;
      (void)symbolic_fields;
    }


    template <std::size_t I = 0,
              typename SDNumberType,
              typename... SymbolicOpType>
      static typename std::enable_if <
      I<sizeof...(SymbolicOpType), void>::type
      unpack_print_sd_functor_print_field_args_and_symbolic_fields(
        const typename OpHelper_t<SymbolicOpType...>::field_args_t &field_args,
        const typename OpHelper_t<SymbolicOpType...>::template field_values_t<
          SDNumberType> &          symbolic_fields,
        const SymbolicDecorations &decorator)
    {
      deallog << "Field index  " << dealii::Utilities::to_string(I) << ": "
              << std::get<I>(field_args).as_ascii(decorator) << " -> "
              << std::get<I>(symbolic_fields) << std::endl;

      unpack_print_sd_functor_print_field_args_and_symbolic_fields<
        I + 1,
        SDNumberType,
        SymbolicOpType...>(field_args, symbolic_fields, decorator);
    }


    template <typename... SymbolicOpsSubSpaceFieldSolution,
              typename SymbolicOpSDFunctor,
              typename = typename std::enable_if<
                WeakForms::is_sd_functor_op<SymbolicOpSDFunctor>::value>::type>
    void
    print_sd_functor_print_field_args_and_symbolic_fields_impl(
      const SymbolicOpSDFunctor &           sd_functor,
      const WeakForms::SymbolicDecorations &decorator)
    {
      deallog << "Number of components: "
              << dealii::Utilities::to_string(
                   OpHelper_t<
                     SymbolicOpsSubSpaceFieldSolution...>::get_n_components())
              << std::endl;

      unpack_print_sd_functor_print_field_args_and_symbolic_fields<
        0,
        Differentiation::SD::Expression,
        SymbolicOpsSubSpaceFieldSolution...>(sd_functor.get_field_args(),
                                             sd_functor.get_symbolic_fields(),
                                             decorator);
    }


    template <int dim,
              int spacedim,
              typename... SymbolicOpsSubSpaceFieldSolution>
    void
    print_sd_functor_print_field_args_and_symbolic_fields(
      const SymbolicOp<
        WeakForms::EnergyFunctor<SymbolicOpsSubSpaceFieldSolution...>,
        SymbolicOpCodes::value,
        void,
        Differentiation::SD::Expression,
        WeakForms::internal::DimPack<dim, spacedim>> &energy_functor,
      const WeakForms::SymbolicDecorations &          decorator)
    {
      print_sd_functor_print_field_args_and_symbolic_fields_impl<
        SymbolicOpsSubSpaceFieldSolution...>(energy_functor, decorator);
    }


    template <int dim,
              int spacedim,
              typename TestSpaceOp,
              typename... SymbolicOpsSubSpaceFieldSolution>
    void
    print_sd_functor_print_field_args_and_symbolic_fields(
      const SymbolicOp<
        WeakForms::ResidualViewFunctor<TestSpaceOp,
                                       SymbolicOpsSubSpaceFieldSolution...>,
        SymbolicOpCodes::value,
        void,
        Differentiation::SD::Expression,
        WeakForms::internal::DimPack<dim, spacedim>> &residual_functor,
      const WeakForms::SymbolicDecorations &          decorator)
    {
      print_sd_functor_print_field_args_and_symbolic_fields_impl<
        SymbolicOpsSubSpaceFieldSolution...>(residual_functor, decorator);
    }

  } // namespace Operators
} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE
