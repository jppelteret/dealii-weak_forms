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

#ifndef dealii_weakforms_symbolic_decorations_h
#define dealii_weakforms_symbolic_decorations_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>

#include <weak_forms/config.h>
#include <weak_forms/utilities.h>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Decorations
  {
    enum class SymbolicNameType
    {
      ascii,
      latex
    };
    struct Discretization
    {
      Discretization(const std::string solution_field,
                     const std::string test_function,
                     const std::string trial_solution,
                     const std::string shape_function,
                     const std::string dof_value,
                     const std::string JxW);

      /**
       * Symbol for the solution field
       */
      const std::string solution_field;

      /**
       * Symbol for the test function
       */
      const std::string test_function;

      /**
       * Symbol for the trial solution
       */
      const std::string trial_solution;

      /**
       * Symbol for a shape function
       */
      const std::string shape_function;

      /**
       * Symbol for a degree-of-freedom value
       */
      const std::string dof_value;

      /**
       * Symbol for the integration constant
       */
      const std::string JxW;
    };

    struct DifferentialOperators
    {
      DifferentialOperators(const std::string value,
                            const std::string gradient,
                            const std::string symmetric_gradient,
                            const std::string divergence,
                            const std::string curl,
                            const std::string hessian,
                            const std::string laplacian,
                            const std::string third_derivative);

      /**
       * Symbol for the value of the operand
       */
      const std::string value;

      /**
       * Symbol for the gradient operator
       */
      const std::string gradient;

      /**
       * Symbol for the symmetric gradient operator
       */
      const std::string symmetric_gradient;

      /**
       * Symbol for the divergence operator
       */
      const std::string divergence;

      /**
       * Symbol for the curl operator
       */
      const std::string curl;

      /**
       * Symbol for the hessian
       */
      const std::string hessian;

      /**
       * Symbol for the Laplacian
       */
      const std::string laplacian;

      /**
       * Symbol for third derivative
       */
      const std::string third_derivative;
    };

    struct Geometry
    {
      Geometry(const std::string position,
               const std::string normal,
               const std::string volume,
               const std::string boundary,
               const std::string interface);

      /**
       * Symbol for the spatial position / coordinate
       */
      const std::string position;

      /**
       * Symbol for a surface normal
       */
      const std::string normal;

      /**
       * Symbol for a volume
       */
      const std::string volume;

      /**
       * Symbol for a boundary surface
       */
      const std::string boundary;

      /**
       * Symbol for an internal interface
       */
      const std::string interface;
    };

    struct DifferentialGeometry
    {
      DifferentialGeometry(
        const std::string infinitesimal_element_volume,
        const std::string infinitesimal_element_boundary_area,
        const std::string infinitesimal_element_interface_area);

      /**
       * Symbol for an infinitesimal volume
       */
      const std::string infinitesimal_element_volume;

      /**
       * Symbol for an infinitesimal boundary surface area
       */
      const std::string infinitesimal_element_boundary_area;

      /**
       * Symbol for an infinitesimal internal interface area
       */
      const std::string infinitesimal_element_interface_area;
    };

    struct ContinuumMechanics
    {
      ContinuumMechanics(const std::string free_energy,
                         const std::string stored_energy,
                         const std::string energy_functional,
                         const std::string residual);

      const std::string free_energy;
      const std::string stored_energy;
      const std::string energy_functional;
      const std::string residual;
    };


    Discretization
    make_symbolic_names_discretization(const SymbolicNameType &type);

    DifferentialOperators
    make_symbolic_names_differential_operators(const SymbolicNameType &type);

    Geometry
    make_symbolic_names_geometry(const SymbolicNameType &type);

    DifferentialGeometry
    make_symbolic_names_differential_geometry(const SymbolicNameType &type);

    ContinuumMechanics
    make_symbolic_names_continuum_mechanics(const SymbolicNameType &type);

  } // namespace Decorations


  /**
   * A data structure that defines the labels to be used
   * to construct symbolic variables identifiers.
   *
   * @note It is critical to ensure that the labels are
   * unique. If not then there is the possibility that one
   * can generate conflicting symbolic expressions that
   * will not be detected during their use.
   */
  struct SymbolicNames
  {
    /**
     * Default constructor
     */
    explicit SymbolicNames(
      const Decorations::Discretization        discretization,
      const Decorations::Geometry              geometry,
      const Decorations::DifferentialGeometry  differential_geometry,
      const Decorations::DifferentialOperators differential_operators,
      const Decorations::ContinuumMechanics    continuum_mechanics);

    const Decorations::Discretization        discretization;
    const Decorations::Geometry              geometry;
    const Decorations::DifferentialGeometry  differential_geometry;
    const Decorations::DifferentialOperators differential_operators;
    const Decorations::ContinuumMechanics    continuum_mechanics;
  }; // struct SymbolicNames



  struct SymbolicNamesAscii : public SymbolicNames
  {
    /**
     * Default constructor
     */
    explicit SymbolicNamesAscii(
      const Decorations::Discretization discretization =
        Decorations::make_symbolic_names_discretization(
          Decorations::SymbolicNameType::ascii),
      const Decorations::Geometry geometry =
        Decorations::make_symbolic_names_geometry(
          Decorations::SymbolicNameType::ascii),
      const Decorations::DifferentialGeometry differential_geometry =
        Decorations::make_symbolic_names_differential_geometry(
          Decorations::SymbolicNameType::ascii),
      const Decorations::DifferentialOperators differential_operators =
        Decorations::make_symbolic_names_differential_operators(
          Decorations::SymbolicNameType::ascii),
      const Decorations::ContinuumMechanics continuum_mechanics =
        Decorations::make_symbolic_names_continuum_mechanics(
          Decorations::SymbolicNameType::ascii));
  }; // struct SymbolicNamesAscii

  struct SymbolicNamesLaTeX : public SymbolicNames
  {
    /**
     * Default constructor
     */
    explicit SymbolicNamesLaTeX(
      const Decorations::Discretization discretization =
        Decorations::make_symbolic_names_discretization(
          Decorations::SymbolicNameType::latex),
      const Decorations::Geometry geometry =
        Decorations::make_symbolic_names_geometry(
          Decorations::SymbolicNameType::latex),
      const Decorations::DifferentialGeometry differential_geometry =
        Decorations::make_symbolic_names_differential_geometry(
          Decorations::SymbolicNameType::latex),
      const Decorations::DifferentialOperators differential_operators =
        Decorations::make_symbolic_names_differential_operators(
          Decorations::SymbolicNameType::latex),
      const Decorations::ContinuumMechanics continuum_mechanics =
        Decorations::make_symbolic_names_continuum_mechanics(
          Decorations::SymbolicNameType::latex));
  }; // struct SymbolicNamesLaTeX


  /**
   * A class to do all decorations
   *
   */
  struct SymbolicDecorations
  {
    SymbolicDecorations(
      const SymbolicNamesAscii &naming_ascii = SymbolicNamesAscii(),
      const SymbolicNamesLaTeX &naming_latex = SymbolicNamesLaTeX());

    const SymbolicNamesAscii &
    get_naming_ascii() const
    {
      return naming_ascii;
    }

    const SymbolicNamesLaTeX &
    get_naming_latex() const
    {
      return naming_latex;
    }

    std::string
    make_position_dependent_symbol_ascii(const std::string &symbol) const
    {
      const auto &naming = get_naming_ascii().geometry;
      return symbol + "(" + naming.position + ")";
    }

    std::string
    make_position_dependent_symbol_latex(const std::string &symbol) const
    {
      const auto &naming = get_naming_latex().geometry;
      return Utilities::LaTeX::decorate_function_with_arguments(
        symbol, naming.position);
    }

    std::string
    make_time_indexed_symbol_ascii(const std::string &symbol,
                                   const std::size_t  time_index) const
    {
      if (time_index == 0)
        return symbol;
      else
        return symbol + "_" + dealii::Utilities::to_string(time_index);
    }

    std::string
    make_time_indexed_symbol_latex(const std::string &symbol,
                                   const std::size_t  time_index) const
    {
      if (time_index == 0)
        return symbol;
      else
        return symbol + "_{t-" + dealii::Utilities::to_string(time_index) + "}";
    }

    template <typename Operand>
    std::string
    symbolic_op_operand_as_ascii(const Operand &operand) const
    {
      const SymbolicDecorations &decorator = *this;
      const std::string          field     = operand.get_field_ascii(decorator);
      if (field == "")
        return operand.get_symbol_ascii(decorator);

      return operand.get_symbol_ascii(decorator) + "{" +
             operand.get_field_ascii(decorator) + "}";
    }

    template <typename Operand>
    std::string
    symbolic_op_operand_as_latex(const Operand &operand) const
    {
      const SymbolicDecorations &decorator = *this;
      const std::string          field     = operand.get_field_latex(decorator);
      if (field == "")
        return operand.get_symbol_latex(decorator);

      return operand.get_symbol_latex(decorator) + "{" +
             operand.get_field_latex(decorator) + "}";
    }

    template <typename Operand>
    std::string
    mathop_symbolic_op_operand_as_ascii(const std::string &prefix,
                                        const Operand &    operand) const
    {
      const SymbolicDecorations &decorator = *this;
      return prefix + "(" + operand.as_ascii(decorator) + ")";
    }

    template <typename Operand>
    std::string
    mathop_symbolic_op_operand_as_latex(const std::string &prefix,
                                        const Operand &    operand) const
    {
      const std::string lbrace = "{";
      const std::string rbrace = "}";

      const SymbolicDecorations &decorator = *this;
      return prefix + lbrace + operand.as_latex(decorator) + rbrace;
    }

    template <typename Operand>
    std::string
    prefixed_parenthesized_symbolic_op_operand_as_ascii(
      const std::string &prefix,
      const Operand &    operand) const
    {
      const SymbolicDecorations &decorator = *this;
      return prefix + "(" + operand.as_ascii(decorator) + ")";
    }

    template <typename Operand>
    std::string
    prefixed_parenthesized_symbolic_op_operand_as_latex(
      const std::string &prefix,
      const Operand &    operand) const
    {
      const SymbolicDecorations &decorator = *this;
      return Utilities::LaTeX::decorate_function_with_arguments(
        prefix, operand.as_ascii(decorator));
    }

    template <typename Operand>
    std::string
    suffixed_braced_superscript_symbolic_op_operand_as_latex(
      const Operand &    operand,
      const std::string &superscript_suffix) const
    {
      const std::string lbrace = Utilities::LaTeX::l_square_brace;
      const std::string rbrace = Utilities::LaTeX::r_square_brace;

      const SymbolicDecorations &decorator = *this;
      return lbrace + operand.as_latex(decorator) + rbrace + "^{" +
             superscript_suffix + "}";
    }

    template <typename Functor>
    std::string
    symbolic_op_functor_as_ascii(const Functor &    functor,
                                 const unsigned int rank) const
    {
      const SymbolicDecorations &decorator = *this;
      if (rank == 0)
        return functor.get_symbol_ascii(decorator);
      else
        {
          const std::string prefix(rank, '<');
          const std::string suffix(rank, '>');
          return prefix + functor.get_symbol_ascii(decorator) + suffix;
        }
    }

    template <typename Functor>
    std::string
    symbolic_op_functor_as_latex(const Functor &    functor,
                                 const unsigned int rank) const
    {
      const SymbolicDecorations &decorator = *this;
      return Utilities::LaTeX::decorate_tensor(
        functor.get_symbol_latex(decorator), rank);
    }

    template <typename... SymbolicOpType>
    std::string
    unary_field_ops_as_ascii(
      const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions) const
    {
      return unpack_unary_field_ops_as_ascii(symbolic_op_field_solutions);
    }

    template <typename... SymbolicOpType>
    std::string
    unary_field_ops_as_latex(
      const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions) const
    {
      return unpack_unary_field_ops_as_latex(symbolic_op_field_solutions);
    }

    template <typename... SymbolicOpType>
    std::string
    differential_expansion_of_unary_field_ops_as_ascii(
      const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions) const
    {
      return unpack_differential_expansion_of_unary_field_ops_as_ascii(
        symbolic_op_field_solutions);
    }

    template <typename... SymbolicOpType>
    std::string
    differential_expansion_of_unary_field_ops_as_latex(
      const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions) const
    {
      return unpack_differential_expansion_of_unary_field_ops_as_latex(
        symbolic_op_field_solutions);
    }

    template <typename UnaryFunctorOp, typename... UnaryFieldOps>
    std::string
    symbolic_op_derivative_as_ascii(
      const UnaryFunctorOp &              unary_functor,
      const std::tuple<UnaryFieldOps...> &symbolic_op_field_solutions) const
    {
      const SymbolicDecorations &decorator = *this;
      constexpr int              n_diff_operations =
        std::tuple_size<std::tuple<UnaryFieldOps...>>::value;

      if (n_diff_operations == 0)
        return unary_functor.as_ascii(decorator);

      // Form the numerator of the differential notation
      std::string out = "d";
      if (n_diff_operations > 1)
        {
          out += dealii::Utilities::to_string(n_diff_operations);
        }
      out += "(" + unary_functor.as_ascii(decorator) + ")";

      // Form the denominator of the differential notation
      out += "/";
      if (n_diff_operations > 1)
        {
          out += "(";
        }
      out += differential_expansion_of_unary_field_ops_as_ascii(
        symbolic_op_field_solutions);
      if (n_diff_operations > 1)
        {
          out += ")";
        }

      return out;
    }

    template <typename UnaryFunctorOp, typename... UnaryFieldOps>
    std::string
    symbolic_op_derivative_as_latex(
      const UnaryFunctorOp &              unary_functor,
      const std::tuple<UnaryFieldOps...> &symbolic_op_field_solutions) const
    {
      const SymbolicDecorations &decorator = *this;
      constexpr int              n_diff_operations =
        std::tuple_size<std::tuple<UnaryFieldOps...>>::value;

      if (n_diff_operations == 0)
        return unary_functor.as_latex(decorator);

      const std::string numerator = Utilities::LaTeX::decorate_differential(
        unary_functor.as_latex(decorator), n_diff_operations);
      const std::string denominator =
        differential_expansion_of_unary_field_ops_as_latex(
          symbolic_op_field_solutions);
      return Utilities::LaTeX::decorate_fraction(numerator, denominator);
    }

    template <typename Functor, typename Infinitesimal>
    std::string
    symbolic_op_integral_as_ascii(
      const Functor &      functor,
      const Infinitesimal &infinitesimal_element) const
    {
      const std::string          prefix("#");
      const std::string          suffix("#");
      const SymbolicDecorations &decorator = *this;

      if (infinitesimal_element.integrate_over_entire_domain())
        {
          return prefix + functor.as_ascii(decorator) + suffix +
                 infinitesimal_element.get_infinitesimal_symbol_ascii(
                   decorator);
        }
      else
        {
          Assert(!infinitesimal_element.get_subdomains().empty(),
                 ExcInternalError());

          // Expand the set of subdomains as a comma separated list
          const auto &      subdomains = infinitesimal_element.get_subdomains();
          const std::string str_subdomains =
            Utilities::get_comma_separated_string_from(subdomains);

          return prefix + functor.as_ascii(decorator) + suffix +
                 infinitesimal_element.get_infinitesimal_symbol_ascii(
                   decorator) +
                 "(" + infinitesimal_element.get_symbol_ascii(decorator) + "=" +
                 str_subdomains + ")";
        }
    }


    // template <typename Functor, typename SubDomainType, template<typename>
    // class Infinitesimal> std::string symbolic_op_integral_as_latex(const
    // Functor &      functor,
    //                            const Infinitesimal<SubDomainType>
    //                            &infinitesimal_element) const
    template <typename Functor, typename Infinitesimal>
    std::string
    symbolic_op_integral_as_latex(
      const Functor &      functor,
      const Infinitesimal &infinitesimal_element) const
    {
      const SymbolicDecorations &decorator = *this;
      if (infinitesimal_element.integrate_over_entire_domain())
        {
          return Utilities::LaTeX::decorate_integral(
            functor.as_latex(decorator),
            infinitesimal_element.get_infinitesimal_symbol_latex(decorator));
        }
      else
        {
          Assert(!infinitesimal_element.get_subdomains().empty(),
                 ExcInternalError());

          // Expand the set of subdomains as a comma separated list
          const auto &      subdomains = infinitesimal_element.get_subdomains();
          const std::string str_subdomains =
            Utilities::get_comma_separated_string_from(subdomains);

          return Utilities::LaTeX::decorate_integral(
            functor.as_latex(decorator),
            infinitesimal_element.get_infinitesimal_symbol_latex(decorator),
            infinitesimal_element.get_symbol_ascii(decorator) + "=" +
              str_subdomains);
        }
    }


    /**
     *
     *
     * @param op A string that symbolises the operator that acts on the @p operand.
     * @param operand
     * @return std::string
     */
    std::string
    decorate_with_operator_ascii(const std::string &op,
                                 const std::string &operand) const
    {
      if (op == "")
        return operand;

      return op + "(" + operand + ")";
    }


    /**
     *
     *
     * @param op A string that symbolises the operator that acts on the @p operand.
     * @param operand
     * @return std::string
     */
    std::string
    decorate_with_operator_latex(const std::string &op,
                                 const std::string &operand) const
    {
      if (op == "")
        return operand;

      return Utilities::LaTeX::decorate_function_with_arguments(op, operand);
    }

    const SymbolicNamesAscii naming_ascii;
    const SymbolicNamesLaTeX naming_latex;

  private:
    template <std::size_t I = 0, typename... SymbolicOpType>
      inline typename std::enable_if <
      I<sizeof...(SymbolicOpType), std::string>::type
      unpack_unary_field_ops_as_ascii(
        const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions) const
    {
      if (I < sizeof...(SymbolicOpType) - 1)
        return std::get<I>(symbolic_op_field_solutions).as_ascii(*this) + ", " +
               unpack_unary_field_ops_as_ascii<I + 1, SymbolicOpType...>(
                 symbolic_op_field_solutions);
      else
        return std::get<I>(symbolic_op_field_solutions).as_ascii(*this);
    }

    // unary_field_ops_as_ascii(): End point
    template <std::size_t I = 0, typename... SymbolicOpType>
    inline
      typename std::enable_if<I == sizeof...(SymbolicOpType), std::string>::type
      unpack_unary_field_ops_as_ascii(
        const std::tuple<SymbolicOpType...> &symbolic_op_field_solution) const
    {
      // Do nothing
      (void)symbolic_op_field_solution;
      return "";
    }

    template <std::size_t I = 0, typename... SymbolicOpType>
      inline typename std::enable_if <
      I<sizeof...(SymbolicOpType), std::string>::type
      unpack_unary_field_ops_as_latex(
        const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions) const
    {
      if (I < sizeof...(SymbolicOpType) - 1)
        return std::get<I>(symbolic_op_field_solutions).as_latex(*this) + ", " +
               unpack_unary_field_ops_as_latex<I + 1, SymbolicOpType...>(
                 symbolic_op_field_solutions);
      else
        return std::get<I>(symbolic_op_field_solutions).as_latex(*this);
    }

    // unary_field_ops_as_latex(): End point
    template <std::size_t I = 0, typename... SymbolicOpType>
    inline
      typename std::enable_if<I == sizeof...(SymbolicOpType), std::string>::type
      unpack_unary_field_ops_as_latex(
        const std::tuple<SymbolicOpType...> &symbolic_op_field_solution) const
    {
      // Do nothing
      (void)symbolic_op_field_solution;
      return "";
    }


    template <std::size_t I = 0, typename... SymbolicOpType>
    inline typename std::enable_if<(sizeof...(SymbolicOpType) >= 2) &&
                                     (I < sizeof...(SymbolicOpType) - 1),
                                   std::string>::type
    unpack_differential_expansion_of_unary_field_ops_as_ascii(
      const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions) const
    {
      const auto &lhs_op = std::get<I>(symbolic_op_field_solutions);
      const auto &rhs_op = std::get<I + 1>(symbolic_op_field_solutions);

      // If either operator is a scalar operator, then we just separate
      // the two differential operations. If none are scalar, then we
      // use some tensor outer product notation as the divider.
      const std::string symbol_outer_product =
        (lhs_op.rank == 0 || rhs_op.rank == 0 ? " " : " x ");

      return "d" + std::get<I>(symbolic_op_field_solutions).as_ascii(*this) +
             symbol_outer_product +
             unpack_differential_expansion_of_unary_field_ops_as_ascii<
               I + 1,
               SymbolicOpType...>(symbolic_op_field_solutions);
    }

    template <std::size_t I = 0, typename... SymbolicOpType>
    inline typename std::enable_if<((sizeof...(SymbolicOpType) >= 2) &&
                                    (I == sizeof...(SymbolicOpType) - 1)) ||
                                     ((sizeof...(SymbolicOpType) < 2) &&
                                      (I < sizeof...(SymbolicOpType))),
                                   std::string>::type
    unpack_differential_expansion_of_unary_field_ops_as_ascii(
      const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions) const
    {
      // Only a single element to fetch
      return "d" + std::get<I>(symbolic_op_field_solutions).as_ascii(*this);
    }

    // unary_field_ops_as_ascii(): End point
    template <std::size_t I = 0, typename... SymbolicOpType>
    inline
      typename std::enable_if<I == sizeof...(SymbolicOpType), std::string>::type
      unpack_differential_expansion_of_unary_field_ops_as_ascii(
        const std::tuple<SymbolicOpType...> &symbolic_op_field_solution) const
    {
      // Do nothing
      (void)symbolic_op_field_solution;
      return "";
    }

    template <std::size_t I = 0, typename... SymbolicOpType>
    inline typename std::enable_if<(sizeof...(SymbolicOpType) >= 2) &&
                                     (I < sizeof...(SymbolicOpType) - 1),
                                   std::string>::type
    unpack_differential_expansion_of_unary_field_ops_as_latex(
      const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions) const
    {
      const auto &lhs_op = std::get<I>(symbolic_op_field_solutions);
      const auto &rhs_op = std::get<I + 1>(symbolic_op_field_solutions);

      // If either operator is a scalar operator, then we just separate
      // the two differential operations. If none are scalar, then we
      // use some tensor outer product notation as the divider.
      const std::string symbol_outer_product =
        Utilities::LaTeX::get_symbol_outer_product(lhs_op.rank, rhs_op.rank);

      return Utilities::LaTeX::decorate_differential(
        std::get<I>(symbolic_op_field_solutions).as_latex(*this) +
        symbol_outer_product +
        unpack_differential_expansion_of_unary_field_ops_as_latex<
          I + 1,
          SymbolicOpType...>(symbolic_op_field_solutions));
    }

    template <std::size_t I = 0, typename... SymbolicOpType>
    inline typename std::enable_if<((sizeof...(SymbolicOpType) >= 2) &&
                                    (I == sizeof...(SymbolicOpType) - 1)) ||
                                     ((sizeof...(SymbolicOpType) < 2) &&
                                      (I < sizeof...(SymbolicOpType))),
                                   std::string>::type
    unpack_differential_expansion_of_unary_field_ops_as_latex(
      const std::tuple<SymbolicOpType...> &symbolic_op_field_solutions) const
    {
      // Only a single element to fetch
      return Utilities::LaTeX::decorate_differential(
        std::get<I>(symbolic_op_field_solutions).as_latex(*this));
    }

    // unary_field_ops_as_latex(): End point
    template <std::size_t I = 0, typename... SymbolicOpType>
    inline
      typename std::enable_if<I == sizeof...(SymbolicOpType), std::string>::type
      unpack_differential_expansion_of_unary_field_ops_as_latex(
        const std::tuple<SymbolicOpType...> &symbolic_op_field_solution) const
    {
      // Do nothing
      (void)symbolic_op_field_solution;
      return "";
    }
  };

} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_symbolic_decorations_h
