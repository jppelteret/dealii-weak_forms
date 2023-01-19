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

#ifndef dealii_weakforms_symbolic_integral_h
#define dealii_weakforms_symbolic_integral_h

#include <deal.II/base/config.h>

#include <weak_forms/config.h>

// TODO: Move FeValuesViews::[Scalar/Vector/...]::Output<> into another header??
// #include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>

#include <weak_forms/numbers.h>
#include <weak_forms/symbolic_decorations.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/types.h>

#include <functional>
#include <memory>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  namespace internal
  {
    // It looks like the only way to implement a general predicate that
    // inevitably has some templates associated with it is through type erasure.
    // https://stackoverflow.com/a/34815953
    // https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Type_Erasure
    // https://davekilian.com/cpp-type-erasure.html

    template <bool isCellPredicate>
    struct PredicateKeeper
    {
      template <int dim, int spacedim>
      using tria_iterator = typename std::conditional<
        isCellPredicate,
        typename Triangulation<dim, spacedim>::cell_iterator,
        typename Triangulation<dim, spacedim>::face_iterator>::type;

      template <int dim, int spacedim>
      using dofhandler_iterator = typename std::conditional<
        isCellPredicate,
        typename DoFHandler<dim, spacedim>::cell_iterator,
        typename DoFHandler<dim, spacedim>::face_iterator>::type;

      template <int dim, int spacedim>
      using TriangulationPredicateFunctionType =
        std::function<bool(const tria_iterator<dim, spacedim> &)>;

      template <int dim, int spacedim>
      using DoFHandlerPredicateFunctionType =
        std::function<bool(const dofhandler_iterator<dim, spacedim> &)>;

      template <class PredicateType>
      PredicateKeeper(const PredicateType &predicate)
        : tria_predicate_11([predicate](const tria_iterator<1, 1> &it)
                            { return predicate(it); })
        , tria_predicate_12([predicate](const tria_iterator<1, 2> &it)
                            { return predicate(it); })
        , tria_predicate_13([predicate](const tria_iterator<1, 3> &it)
                            { return predicate(it); })
        , tria_predicate_22([predicate](const tria_iterator<2, 2> &it)
                            { return predicate(it); })
        , tria_predicate_23([predicate](const tria_iterator<2, 2> &it)
                            { return predicate(it); })
        , tria_predicate_33([predicate](const tria_iterator<3, 3> &it)
                            { return predicate(it); })
        , dofhandler_predicate_11(
            [predicate](const dofhandler_iterator<1, 1> &it)
            { return predicate(it); })
        , dofhandler_predicate_12(
            [predicate](const dofhandler_iterator<1, 2> &it)
            { return predicate(it); })
        , dofhandler_predicate_13(
            [predicate](const dofhandler_iterator<1, 3> &it)
            { return predicate(it); })
        , dofhandler_predicate_22(
            [predicate](const dofhandler_iterator<2, 2> &it)
            { return predicate(it); })
        , dofhandler_predicate_23(
            [predicate](const dofhandler_iterator<2, 3> &it)
            { return predicate(it); })
        , dofhandler_predicate_33(
            [predicate](const dofhandler_iterator<3, 3> &it)
            { return predicate(it); })
      {}

      template <class Iterator>
      bool
      operator()(const Iterator &iterator) const
      {
        return apply(iterator);
      }

    private:
      const TriangulationPredicateFunctionType<1, 1> tria_predicate_11;
      const TriangulationPredicateFunctionType<1, 2> tria_predicate_12;
      const TriangulationPredicateFunctionType<1, 3> tria_predicate_13;
      const TriangulationPredicateFunctionType<2, 2> tria_predicate_22;
      const TriangulationPredicateFunctionType<2, 3> tria_predicate_23;
      const TriangulationPredicateFunctionType<3, 3> tria_predicate_33;
      const DoFHandlerPredicateFunctionType<1, 1>    dofhandler_predicate_11;
      const DoFHandlerPredicateFunctionType<1, 2>    dofhandler_predicate_12;
      const DoFHandlerPredicateFunctionType<1, 3>    dofhandler_predicate_13;
      const DoFHandlerPredicateFunctionType<2, 2>    dofhandler_predicate_22;
      const DoFHandlerPredicateFunctionType<2, 3>    dofhandler_predicate_23;
      const DoFHandlerPredicateFunctionType<3, 3>    dofhandler_predicate_33;

      bool
      apply(const tria_iterator<1, 1> &iterator) const
      {
        return tria_predicate_11(iterator);
      }

      bool
      apply(const tria_iterator<1, 2> &iterator) const
      {
        return tria_predicate_12(iterator);
      }

      bool
      apply(const tria_iterator<1, 3> &iterator) const
      {
        return tria_predicate_13(iterator);
      }

      bool
      apply(const tria_iterator<2, 2> &iterator) const
      {
        return tria_predicate_22(iterator);
      }

      bool
      apply(const tria_iterator<2, 3> &iterator) const
      {
        return tria_predicate_23(iterator);
      }

      bool
      apply(const tria_iterator<3, 3> &iterator) const
      {
        return tria_predicate_33(iterator);
      }

      bool
      apply(const dofhandler_iterator<1, 1> &iterator) const
      {
        return dofhandler_predicate_11(iterator);
      }

      bool
      apply(const dofhandler_iterator<1, 2> &iterator) const
      {
        return dofhandler_predicate_12(iterator);
      }

      bool
      apply(const dofhandler_iterator<1, 3> &iterator) const
      {
        return dofhandler_predicate_13(iterator);
      }

      bool
      apply(const dofhandler_iterator<2, 2> &iterator) const
      {
        return dofhandler_predicate_22(iterator);
      }

      bool
      apply(const dofhandler_iterator<2, 3> &iterator) const
      {
        return dofhandler_predicate_23(iterator);
      }

      bool
      apply(const dofhandler_iterator<3, 3> &iterator) const
      {
        return dofhandler_predicate_33(iterator);
      }
    };


    template <bool isCellPredicate>
    class PredicateHolder
    {
      // TODO: Store this in a vector, so that it can act like a
      // FilteredIterator.
      std::unique_ptr<PredicateKeeper<isCellPredicate>> predicates;

    public:
      template <class PredicateType>
      PredicateHolder(const PredicateType &predicate)
        : predicates(new PredicateKeeper<isCellPredicate>(predicate))
      {}

      template <class Iterator>
      bool
      operator()(const Iterator &iterator) const
      {
        Assert(predicates, ExcNotInitialized());
        return (*predicates)(iterator);
      }
    };

  } // namespace internal


  /**
   * @brief A base class for other objects that represent (sub)domains of integration.
   *
   * @tparam SubDomainType The value type for the subdomain to be considered as
   * a part of the set of finite elements for integration.
   */
  template <typename PredicateType,
            bool isCellPredicate,
            typename SubDomainType>
  class Integral
  {
  public:
    template <typename ScalarType>
    using value_type = double;

    using subdomain_t = SubDomainType;
    using PrintFunctionType =
      std::function<std::string(const SymbolicDecorations &decorator)>;

    Integral(const PredicateType &    predicate,
             const PrintFunctionType &subdomain_as_ascii,
             const PrintFunctionType &subdomain_as_latex)
      : predicate(std::make_shared<internal::PredicateHolder<isCellPredicate>>(
          predicate))
      , subdomain_as_ascii(subdomain_as_ascii)
      , subdomain_as_latex(subdomain_as_latex)
    {}

    Integral(const PredicateType &predicate,
             const std::string &  subdomain_as_ascii,
             const std::string &  subdomain_as_latex)
      : Integral(
          predicate,
          [subdomain_as_ascii](const SymbolicDecorations &)
          { return subdomain_as_ascii; },
          [subdomain_as_latex](const SymbolicDecorations &)
          { return subdomain_as_latex; })
    {}

    Integral(const std::set<SubDomainType> &subdomains)
    {
      if (!subdomains.empty())
        {
          predicate =
            std::make_shared<internal::PredicateHolder<isCellPredicate>>(
              PredicateType(subdomains));

          subdomain_as_ascii = PrintFunctionType(
            [subdomains](const SymbolicDecorations &)
            {
              // Expand the set of subdomains as a comma separated list
              return Utilities::get_comma_separated_string_from(subdomains);
            });

          subdomain_as_latex = PrintFunctionType(
            [subdomains](const SymbolicDecorations &)
            {
              // Expand the set of subdomains as a comma separated list
              return Utilities::get_comma_separated_string_from(subdomains);
            });
        }
    }

    bool
    integrate_over_entire_domain() const
    {
      return !predicate;
    }

    // ----  Ascii ----

    std::string
    as_ascii(const SymbolicDecorations &decorator) const
    {
      return get_infinitesimal_symbol_ascii(decorator);
    }


    std::string
    get_subdomain_as_ascii(const SymbolicDecorations &decorator) const
    {
      if (!subdomain_as_ascii)
        return "";

      return subdomain_as_ascii(decorator);
    }

    virtual std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const = 0;

    virtual std::string
    get_infinitesimal_symbol_ascii(
      const SymbolicDecorations &decorator) const = 0;

    // ---- LaTeX ----

    std::string
    as_latex(const SymbolicDecorations &decorator) const
    {
      return get_infinitesimal_symbol_latex(decorator);
    }

    std::string
    get_subdomain_as_latex(const SymbolicDecorations &decorator) const
    {
      if (!subdomain_as_latex)
        return "";

      return subdomain_as_latex(decorator);
    }

    virtual std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const = 0;

    virtual std::string
    get_infinitesimal_symbol_latex(
      const SymbolicDecorations &decorator) const = 0;

  protected:
    std::shared_ptr<internal::PredicateHolder<isCellPredicate>> predicate;

    PrintFunctionType subdomain_as_ascii;
    PrintFunctionType subdomain_as_latex;

    template <typename IteratorType>
    bool
    integrate_on_subdomain(const IteratorType &iterator) const
    {
      if (integrate_over_entire_domain())
        return true;

      Assert(predicate, ExcNotInitialized());

      return (*predicate)(iterator);
    }
  };



  /**
   * @brief A class that encapsulates the subset of elements that are to be considered as part of a volume integral.
   *
   * This class is not typically created directly by a user, but rather would be
   * automatically generated by a form or function integrator.
   */
  template <typename PredicateType = types::default_volume_integral_predicate_t>
  class VolumeIntegral
    : public Integral<PredicateType, true, dealii::types::material_id>
  {
    using Base = Integral<PredicateType, true, dealii::types::material_id>;

  public:
    using subdomain_t       = typename Base::subdomain_t;
    using PrintFunctionType = typename Base::PrintFunctionType;

    VolumeIntegral(const PredicateType &    predicate,
                   const PrintFunctionType &subdomain_as_ascii,
                   const PrintFunctionType &subdomain_as_latex)
      : Base(predicate, subdomain_as_ascii, subdomain_as_latex)
    {}

    VolumeIntegral(const PredicateType &predicate,
                   const std::string &  subdomain_as_ascii,
                   const std::string &  subdomain_as_latex)
      : Base(predicate, subdomain_as_ascii, subdomain_as_latex)
    {}

    VolumeIntegral(const std::set<subdomain_t> &subregions)
      : Base(subregions)
    {}

    VolumeIntegral(const subdomain_t &subregion)
      : VolumeIntegral(std::set<subdomain_t>{subregion})
    {}

    VolumeIntegral()
      : VolumeIntegral(std::set<subdomain_t>{})
    {}

    std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const override
    {
      const auto &naming = decorator.get_naming_ascii().geometry;
      return naming.volume;
    }

    std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      const auto &naming = decorator.get_naming_latex().geometry;
      return naming.volume;
    }

    std::string
    get_infinitesimal_symbol_ascii(
      const SymbolicDecorations &decorator) const override
    {
      const auto &naming = decorator.get_naming_ascii().differential_geometry;
      return naming.infinitesimal_element_volume;
    }

    std::string
    get_infinitesimal_symbol_latex(
      const SymbolicDecorations &decorator) const override
    {
      const auto &naming = decorator.get_naming_latex().differential_geometry;
      return naming.infinitesimal_element_volume;
    }

    template <typename CellIteratorType>
    bool
    integrate_on_cell(const CellIteratorType &cell) const
    {
      return this->integrate_on_subdomain(cell);
    }

    // Methods to promote this class to a SymbolicOp

    template <typename ScalarType = double, typename Integrand>
    auto
    integrate(const Integrand &integrand) const;
  };



  /**
   * @brief A class that encapsulates the subset of elements that are to be considered as part of a boundary integral.
   *
   * This class is not typically created directly by a user, but rather would be
   * automatically generated by a form or function integrator.
   */
  template <typename PredicateType =
              types::default_boundary_integral_predicate_t>
  class BoundaryIntegral
    : public Integral<PredicateType, false, dealii::types::boundary_id>
  {
    using Base = Integral<PredicateType, false, dealii::types::boundary_id>;

  public:
    using subdomain_t       = typename Base::subdomain_t;
    using PrintFunctionType = typename Base::PrintFunctionType;

    BoundaryIntegral(const PredicateType &    predicate,
                     const PrintFunctionType &subdomain_as_ascii,
                     const PrintFunctionType &subdomain_as_latex)
      : Base(predicate, subdomain_as_ascii, subdomain_as_latex)
    {}

    BoundaryIntegral(const PredicateType &predicate,
                     const std::string &  subdomain_as_ascii,
                     const std::string &  subdomain_as_latex)
      : Base(predicate, subdomain_as_ascii, subdomain_as_latex)
    {}

    BoundaryIntegral(const std::set<subdomain_t> &boundaries)
      : Base(boundaries)
    {}

    BoundaryIntegral(const subdomain_t &boundary)
      : BoundaryIntegral(std::set<subdomain_t>{boundary})
    {}

    BoundaryIntegral()
      : BoundaryIntegral(std::set<subdomain_t>{})
    {}

    std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const override
    {
      const auto &naming = decorator.get_naming_ascii().geometry;
      return naming.boundary;
    }

    std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      const auto &naming = decorator.get_naming_latex().geometry;
      return naming.boundary;
    }

    std::string
    get_infinitesimal_symbol_ascii(
      const SymbolicDecorations &decorator) const override
    {
      const auto &naming = decorator.get_naming_ascii().differential_geometry;
      return naming.infinitesimal_element_boundary_area;
    }

    std::string
    get_infinitesimal_symbol_latex(
      const SymbolicDecorations &decorator) const override
    {
      const auto &naming = decorator.get_naming_latex().differential_geometry;
      return naming.infinitesimal_element_boundary_area;
    }

    template <typename CellIteratorType>
    bool
    integrate_on_face(const CellIteratorType &cell,
                      const unsigned int      face) const
    {
      if (!cell->face(face)->at_boundary())
        return false;

      return this->integrate_on_subdomain(cell->face(face));
    }

    // Methods to promote this class to a SymbolicOp

    template <typename ScalarType = double, typename Integrand>
    auto
    integrate(const Integrand &integrand) const;
  };



  /**
   * @brief A class that encapsulates the subset of elements that are to be considered as part of an interface integral.
   *
   * This class is not typically created directly by a user, but rather would be
   * automatically generated by a form or function integrator.
   */
  template <typename PredicateType =
              types::default_interface_integral_predicate_t>
  class InterfaceIntegral
    : public Integral<PredicateType, false, dealii::types::manifold_id>
  {
    using Base = Integral<PredicateType, false, dealii::types::manifold_id>;

  public:
    using subdomain_t       = typename Base::subdomain_t;
    using PrintFunctionType = typename Base::PrintFunctionType;

    InterfaceIntegral(const PredicateType &    predicate,
                      const PrintFunctionType &subdomain_as_ascii,
                      const PrintFunctionType &subdomain_as_latex)
      : Base(predicate, subdomain_as_ascii, subdomain_as_latex)
    {}

    InterfaceIntegral(const PredicateType &predicate,
                      const std::string &  subdomain_as_ascii,
                      const std::string &  subdomain_as_latex)
      : Base(predicate, subdomain_as_ascii, subdomain_as_latex)
    {}

    InterfaceIntegral(const std::set<subdomain_t> interfaces)
      : Base(interfaces)
    {}

    InterfaceIntegral(const subdomain_t &interface)
      : InterfaceIntegral(std::set<subdomain_t>{interface})
    {}

    InterfaceIntegral()
      : InterfaceIntegral(std::set<subdomain_t>{})
    {}

    std::string
    get_symbol_ascii(const SymbolicDecorations &decorator) const override
    {
      const auto &naming = decorator.get_naming_ascii().geometry;
      return naming.interface;
    }

    std::string
    get_symbol_latex(const SymbolicDecorations &decorator) const override
    {
      const auto &naming = decorator.get_naming_latex().geometry;
      return naming.interface;
    }

    std::string
    get_infinitesimal_symbol_ascii(
      const SymbolicDecorations &decorator) const override
    {
      const auto &naming = decorator.get_naming_ascii().differential_geometry;
      return naming.infinitesimal_element_interface_area;
    }

    std::string
    get_infinitesimal_symbol_latex(
      const SymbolicDecorations &decorator) const override
    {
      const auto &naming = decorator.get_naming_latex().differential_geometry;
      return naming.infinitesimal_element_interface_area;
    }

    template <typename CellIteratorType>
    bool
    integrate_on_face(const CellIteratorType &cell,
                      const unsigned int      face,
                      const unsigned int      neighbour_face) const
    {
      (void)neighbour_face;
      if (cell->face(face)->at_boundary())
        return false;

      return this->integrate_on_subdomain(cell->face(face));
    }

    // Methods to promote this class to a SymbolicOp

    template <typename ScalarType = double, typename Integrand>
    auto
    integrate(const Integrand &integrand) const;
  };



  // class CurveIntegral : public Integral
  // {
  // public:
  //   CurveIntegral(const SymbolicDecorations &decorator =
  //   SymbolicDecorations())
  //     : Integral(decorator.naming_ascii.infinitesimal_element_curve_length,
  //                    decorator.naming_latex.infinitesimal_element_curve_length,
  //                    decorator)
  //   {}
  // };

} // namespace WeakForms



/* ======================== Convenience functions ======================== */



namespace WeakForms
{
  /**
   * @brief A convenience function that returns an operator that represents an integral operation.
   *
   * This function is not typically called directly by a user, but rather would
   * be used by a form or function integrator.
   *
   * @param integrand The quantity or set of operations that are to be integrated.
   * @param integral The domain over which the @p integrand is to be integrated.
   *
   * \ingroup convenience_functions
   */
  template <typename ScalarType = double,
            typename Integrand,
            typename IntegralType,
            typename = typename std::enable_if<is_valid_integration_domain<
              typename std::decay<IntegralType>::type>::value>::type>
#ifndef DOXYGEN
  WeakForms::Operators::SymbolicOp<IntegralType,
                                   WeakForms::Operators::SymbolicOpCodes::value,
                                   ScalarType,
                                   Integrand>
#else
  auto
#endif // DOXYGEN
  integrate(const Integrand &integrand, const IntegralType &integral)
  {
    return integral.template integrate<ScalarType>(integrand);
  }

  // template <typename ScalarType = double,
  //           typename Integrand,
  //           typename IntegralType,
  //           typename = typename std::enable_if<
  //             WeakForms::is_unary_integral_op<IntegralType>::value>::type>
  // auto
  // integrate(const Integrand &integrand, const IntegralType &integral, const
  // std::set<typename IntegralType::subdomain_t> &subdomains)
  // {
  //   return value(integral, integrand);
  // }

} // namespace WeakForms



/* ================== Specialization of unary operators ================== */



namespace WeakForms
{
  namespace Operators
  {
    /**
     * Get the weighted Jacobians for numerical integration
     *
     * @tparam dim
     * @tparam spacedim
     */
    template <typename ScalarType_,
              typename IntegralType_,
              typename IntegrandType_>
    class SymbolicOp<IntegralType_,
                     SymbolicOpCodes::value,
                     ScalarType_,
                     IntegrandType_>
    {
      static_assert(!is_integral_op<IntegrandType_>::value,
                    "Cannot integrate an integral!");

    public:
      using ScalarType    = ScalarType_;
      using IntegralType  = IntegralType_;
      using IntegrandType = IntegrandType_;

      template <typename ScalarType2>
      using value_type =
        typename IntegralType::template value_type<ScalarType2>;

      template <typename ScalarType2>
      using return_type = std::vector<value_type<ScalarType2>>;

      template <typename ResultScalarType, std::size_t width>
      using vectorized_value_type = typename numbers::VectorizedValue<
        value_type<ResultScalarType>>::template type<width>;

      template <typename ResultScalarType, std::size_t width>
      using vectorized_return_type = typename numbers::VectorizedValue<
        value_type<ResultScalarType>>::template type<width>;

      static const int rank = 0;

      // static const enum SymbolicOpCodes op_code = SymbolicOpCodes::value;

      explicit SymbolicOp(const IntegralType & integral_operation,
                          const IntegrandType &integrand)
        : integral_operation(integral_operation)
        , integrand(integrand)
      {}

      bool
      integrate_over_entire_domain() const
      {
        return integral_operation.integrate_over_entire_domain();
      }

      std::string
      as_ascii(const SymbolicDecorations &decorator) const
      {
        return decorator.symbolic_op_integral_as_ascii(integrand,
                                                       integral_operation);
      }

      std::string
      as_latex(const SymbolicDecorations &decorator) const
      {
        const bool use_integral_notation =
          (decorator.get_formatting_latex().get_integral_format() ==
           FormattingLaTeX::IntegralFormat::standard_notation);

        if (use_integral_notation)
          {
            return decorator.symbolic_op_standard_integral_as_latex(
              integrand, integral_operation);
          }
        else
          {
            if (is_volume_integral<IntegralType>::value)
              {
                return decorator.symbolic_op_bilinear_form_integral_as_latex(
                  integrand,
                  integral_operation,
                  FormattingLaTeX::IntegralType::volume_integral);
              }
            else if (is_boundary_integral<IntegralType>::value)
              {
                return decorator.symbolic_op_bilinear_form_integral_as_latex(
                  integrand,
                  integral_operation,
                  FormattingLaTeX::IntegralType::boundary_integral);
              }
            else if (is_interface_integral<IntegralType>::value)
              {
                return decorator.symbolic_op_bilinear_form_integral_as_latex(
                  integrand,
                  integral_operation,
                  FormattingLaTeX::IntegralType::interface_integral);
              }
            else
              {
                AssertThrow(false, ExcNotImplemented());
                return "";
              }
          }
      }

      std::string
      get_subdomain_as_ascii(const SymbolicDecorations &decorator) const
      {
        return integral_operation.get_subdomain_as_ascii(decorator);
      }

      std::string
      get_subdomain_as_latex(const SymbolicDecorations &decorator) const
      {
        return integral_operation.get_subdomain_as_latex(decorator);
      }

      // ===== Section: Construct assembly operation =====

      const IntegralType &
      get_integral_operation() const
      {
        return integral_operation;
      }

      const IntegrandType &
      get_integrand() const
      {
        return integrand;
      }

      // ===== Section: Perform actions =====

      UpdateFlags
      get_update_flags() const
      {
        return get_integrand().get_update_flags() |
               UpdateFlags::update_JxW_values;
      }

      /**
       * Return all JxW values at all quadrature points
       */
      template <typename ScalarType2, int dim, int spacedim>
      const return_type<ScalarType2> &
      operator()(const FEValuesBase<dim, spacedim> &fe_values) const
      {
        return fe_values.get_JxW_values();
      }

      template <typename ScalarType2, int dim, int spacedim>
      const return_type<ScalarType2> &
      operator()(
        const FEInterfaceValues<dim, spacedim> &fe_interface_values) const
      {
        return fe_interface_values.get_JxW_values();
      }

      template <typename ResultScalarType,
                std::size_t width,
                int         dim,
                int         spacedim>
      vectorized_return_type<ResultScalarType, width>
      operator()(const FEValuesBase<dim, spacedim> & fe_values,
                 const types::vectorized_qp_range_t &q_point_range) const
      {
        vectorized_return_type<ResultScalarType, width> out;
        Assert(q_point_range.size() <= width,
               ExcIndexRange(q_point_range.size(), 0, width));

        DEAL_II_OPENMP_SIMD_PRAGMA
        for (unsigned int i = 0; i < q_point_range.size(); ++i)
          if (q_point_range[i] < fe_values.n_quadrature_points)
            numbers::set_vectorized_values(
              out,
              i,
              this->template operator()<ResultScalarType>(fe_values,
                                                          q_point_range[i]));

        return out;
      }

      template <typename ResultScalarType,
                std::size_t width,
                int         dim,
                int         spacedim>
      vectorized_return_type<ResultScalarType, width>
      operator()(const FEInterfaceValues<dim, spacedim> &fe_interface_values,
                 const types::vectorized_qp_range_t &    q_point_range) const
      {
        vectorized_return_type<ResultScalarType, width> out;
        Assert(q_point_range.size() <= width,
               ExcIndexRange(q_point_range.size(), 0, width));

        DEAL_II_OPENMP_SIMD_PRAGMA
        for (unsigned int i = 0; i < q_point_range.size(); ++i)
          if (q_point_range[i] < fe_interface_values.n_quadrature_points)
            numbers::set_vectorized_values(
              out,
              i,
              this->template operator()<ResultScalarType>(fe_interface_values,
                                                          q_point_range[i]));

        return out;
      }

    private:
      const IntegralType  integral_operation;
      const IntegrandType integrand;

      // Return single entry
      template <typename ScalarType2, typename FEValuesType>
      value_type<ScalarType2>
      operator()(const FEValuesType &fe_values,
                 const unsigned int  q_point) const
      {
        Assert(q_point < fe_values.n_quadrature_points,
               ExcIndexRange(q_point, 0, fe_values.n_quadrature_points));

        return fe_values.JxW(q_point);
      }
    };

  } // namespace Operators
} // namespace WeakForms



/* ==================== Class method definitions ==================== */



namespace WeakForms
{
  template <typename PredicateType>
  template <typename ScalarType, typename Integrand>
  DEAL_II_ALWAYS_INLINE inline auto
  WeakForms::VolumeIntegral<PredicateType>::integrate(
    const Integrand &integrand) const
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = VolumeIntegral<PredicateType>;
    using OpType =
      SymbolicOp<Op, SymbolicOpCodes::value, ScalarType, Integrand>;

    const auto &operand = *this;
    return OpType(operand, integrand);
  }


  template <typename PredicateType>
  template <typename ScalarType, typename Integrand>
  DEAL_II_ALWAYS_INLINE inline auto
  WeakForms::BoundaryIntegral<PredicateType>::integrate(
    const Integrand &integrand) const
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = BoundaryIntegral<PredicateType>;
    using OpType =
      SymbolicOp<Op, SymbolicOpCodes::value, ScalarType, Integrand>;

    const auto &operand = *this;
    return OpType(operand, integrand);
  }


  template <typename PredicateType>
  template <typename ScalarType, typename Integrand>
  DEAL_II_ALWAYS_INLINE inline auto
  WeakForms::InterfaceIntegral<PredicateType>::integrate(
    const Integrand &integrand) const
  {
    using namespace WeakForms;
    using namespace WeakForms::Operators;

    using Op = InterfaceIntegral<PredicateType>;
    using OpType =
      SymbolicOp<Op, SymbolicOpCodes::value, ScalarType, Integrand>;

    const auto &operand = *this;
    return OpType(operand, integrand);
  }


  // WeakForms::Operators::SymbolicOp<WeakForms::Integral,
  //                               WeakForms::Operators::SymbolicOpCodes::value>
  // value(const WeakForms::CurveIntegral &operand)
  // {
  //   using namespace WeakForms;
  //   using namespace WeakForms::Operators;

  //   using Op     = Integral;
  //   using OpType = SymbolicOp<Op, SymbolicOpCodes::value>;

  //   return OpType(operand);
  // }

} // namespace WeakForms



/* ==================== Specialization of type traits ==================== */



#ifndef DOXYGEN


namespace WeakForms
{
  template <typename Predicate>
  struct is_valid_integration_domain<VolumeIntegral<Predicate>> : std::true_type
  {};

  template <typename Predicate>
  struct is_valid_integration_domain<BoundaryIntegral<Predicate>>
    : std::true_type
  {};

  template <typename Predicate>
  struct is_valid_integration_domain<InterfaceIntegral<Predicate>>
    : std::true_type
  {};


  // Decorator classes

  template <typename Predicate>
  struct is_volume_integral<VolumeIntegral<Predicate>> : std::true_type
  {};

  template <typename Predicate>
  struct is_boundary_integral<BoundaryIntegral<Predicate>> : std::true_type
  {};

  template <typename Predicate>
  struct is_interface_integral<InterfaceIntegral<Predicate>> : std::true_type
  {};


  // Unary operators

  template <typename ScalarType,
            typename Predicate,
            typename Integrand,
            enum Operators::SymbolicOpCodes OpCode>
  struct is_volume_integral_op<
    Operators::
      SymbolicOp<VolumeIntegral<Predicate>, OpCode, ScalarType, Integrand>>
    : std::true_type
  {};

  template <typename ScalarType,
            typename Predicate,
            typename Integrand,
            enum Operators::SymbolicOpCodes OpCode>
  struct is_boundary_integral_op<
    Operators::
      SymbolicOp<BoundaryIntegral<Predicate>, OpCode, ScalarType, Integrand>>
    : std::true_type
  {};

  template <typename ScalarType,
            typename Predicate,
            typename Integrand,
            enum Operators::SymbolicOpCodes OpCode>
  struct is_interface_integral_op<
    Operators::
      SymbolicOp<InterfaceIntegral<Predicate>, OpCode, ScalarType, Integrand>>
    : std::true_type
  {};

  template <typename T>
  struct is_symbolic_integral_op<
    T,
    typename std::enable_if<is_volume_integral_op<T>::value ||
                            is_boundary_integral_op<T>::value ||
                            is_interface_integral_op<T>::value>::type>
    : std::true_type
  {};

  // I don't know why, but we need this specialisation here.
  template <typename T>
  struct is_integral_op<
    T,
    typename std::enable_if<is_symbolic_integral_op<T>::value>::type>
    : std::true_type
  {};



  // template <typename ScalarType,
  //           typename Integrand,
  //           enum Operators::SymbolicOpCodes OpCode>
  // struct is_symbolic_op<
  //   Operators::SymbolicOp<VolumeIntegral, OpCode, ScalarType, Integrand>>
  //   : std::true_type
  // {};

  // template <typename ScalarType,
  //           typename Integrand,
  //           enum Operators::SymbolicOpCodes OpCode>
  // struct is_symbolic_op<
  //   Operators::SymbolicOp<BoundaryIntegral, OpCode, ScalarType, Integrand>>
  //   : std::true_type
  // {};

  // template <typename ScalarType,
  //           typename Integrand,
  //           enum Operators::SymbolicOpCodes OpCode>
  // struct is_symbolic_op<
  //   Operators::SymbolicOp<InterfaceIntegral, OpCode, ScalarType, Integrand>>
  //   : std::true_type
  // {};

} // namespace WeakForms


#endif // DOXYGEN


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_symbolic_integral_h
