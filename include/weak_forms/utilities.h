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

#ifndef dealii_weakforms_utilities_h
#define dealii_weakforms_utilities_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/template_constraints.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_values_extractors.h>

#include <deal.II/physics/notation.h>

#include <weak_forms/config.h>
#include <weak_forms/template_constraints.h>

#include <iterator>
#include <numeric>
#include <string>
#include <type_traits>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Utilities
  {
    inline std::string
    get_deal_II_prefix()
    {
      return "__DEAL_II__";
    }

    // T must be an iterable type
    template <typename IterableObject>
    std::string
    get_comma_separated_string_from(const IterableObject &t)
    {
      // Expand the object of subdomains as a comma separated list
      // https://stackoverflow.com/a/34076796
      return std::accumulate(std::begin(t),
                             std::end(t),
                             std::string{},
                             [](const std::string &a, const auto &b)
                             {
                               return a.empty() ?
                                        dealii::Utilities::to_string(b) :
                                        a + ',' +
                                          dealii::Utilities::to_string(b);
                             });
    }
    // T must be an iterable type
    template <typename Type>
    std::string
    get_separated_string_from(const Type *const  begin,
                              const Type *const  end,
                              const std::string &seperator)
    {
      // Expand the object of subdomains as a comma separated list
      // https://stackoverflow.com/a/34076796
      return std::accumulate(begin,
                             end,
                             std::string{},
                             [&seperator](const std::string &a, const auto &b)
                             {
                               return a.empty() ?
                                        dealii::Utilities::to_string(b) :
                                        a + seperator +
                                          dealii::Utilities::to_string(b);
                             });
    }


    template <typename T, typename U = void>
    struct ValueHelper;

    template <typename T>
    struct ValueHelper<
      T,
      typename std::enable_if<
        std::is_same<T, typename EnableIfScalar<T>::type>::value>::type>
    {
      static constexpr unsigned int n_components = 1;
      static constexpr unsigned int rank         = 0;
      using extractor_type                       = FEValuesExtractors::Scalar;
    };

    template <int dim, typename T>
    struct ValueHelper<Tensor<1, dim, T>>
    {
      static constexpr unsigned int n_components =
        Tensor<1, dim, T>::n_independent_components;
      static constexpr unsigned int rank = 1;
      using extractor_type               = FEValuesExtractors::Vector;
    };

    template <int rank_, int dim, typename T>
    struct ValueHelper<Tensor<rank_, dim, T>>
    {
      static constexpr unsigned int rank = rank_;
      static constexpr unsigned int n_components =
        Tensor<rank, dim, T>::n_independent_components;
      using extractor_type = FEValuesExtractors::Tensor<rank>;
    };

    template <int rank_, int dim, typename T>
    struct ValueHelper<SymmetricTensor<rank_, dim, T>>
    {
      static constexpr unsigned int rank = rank_;
      static constexpr unsigned int n_components =
        SymmetricTensor<rank, dim, T>::n_independent_components;
      using extractor_type = FEValuesExtractors::SymmetricTensor<rank>;
    };



    /**
     * A small data structure to work out some information that has to do with
     * the contraction of operands during "multiplication" operations
     *
     * @tparam LhsOp
     * @tparam RhsOp
     */
    template <typename LhsOp, typename RhsOp>
    struct FullIndexContraction
    {
      static const int n_contracting_indices =
        (LhsOp::rank < RhsOp::rank ? LhsOp::rank : RhsOp::rank);
      static const int result_rank =
        (LhsOp::rank < RhsOp::rank ? RhsOp::rank - LhsOp::rank :
                                     LhsOp::rank - RhsOp::rank);

      template <int A>
      struct NonNegative
      {
        static_assert(A >= 0, "Non-negative");
        static const int value = A;
      };

      static_assert(NonNegative<n_contracting_indices>::value >= 0,
                    "Number of contracting indices cannot be negative.");
      static_assert(NonNegative<result_rank>::value >= 0,
                    "Cannot have a result with a negative rank.");
    };



    template <typename T, typename U = void>
    struct ConvertNumericToText;


    template <typename ScalarType>
    struct ConvertNumericToText<
      ScalarType,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      static std::string
      to_ascii(const ScalarType value)
      {
        return dealii::Utilities::to_string(value);
      }

      static std::string
      to_latex(const ScalarType value)
      {
        return dealii::Utilities::to_string(value);
      }
    };


    template <int dim, typename ScalarType>
    struct ConvertNumericToText<
      Tensor<0, dim, ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      static std::string
      to_ascii(const Tensor<0, dim, ScalarType> &value)
      {
        // There's only one element; let's treat it like a scalar.
        return ConvertNumericToText<ScalarType>::to_ascii(*value.begin_raw());
      }

      static std::string
      to_latex(const Tensor<0, dim, ScalarType> &value)
      {
        // There's only one element; let's treat it like a scalar.
        return ConvertNumericToText<ScalarType>::to_latex(*value.begin_raw());
      }
    };


    template <int dim, typename ScalarType>
    struct ConvertNumericToText<
      Tensor<1, dim, ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      static std::string
      to_ascii(const Tensor<1, dim, ScalarType> &value)
      {
        std::string out = "[";
        out +=
          get_separated_string_from(value.begin_raw(), value.end_raw(), ",");
        out += "]";
        return out;
      }

      static std::string
      to_latex(const Tensor<1, dim, ScalarType> &value)
      {
        std::string out = "\\begin{pmatrix}";
        out +=
          get_separated_string_from(value.begin_raw(), value.end_raw(), "\\\\");
        out += "\\end{pmatrix}";
        return out;
      }
    };


    template <int dim, typename ScalarType>
    struct ConvertNumericToText<
      Tensor<2, dim, ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      static std::string
      to_ascii(const Tensor<2, dim, ScalarType> &value)
      {
        std::string out = "[";
        for (unsigned int i = 0; i < dim; ++i)
          {
            out += get_separated_string_from(value[i].begin_raw(),
                                             value[i].end_raw(),
                                             ",");
            if (i < dim - 1)
              out += ";";
          }
        out += "]";
        return out;
      }

      static std::string
      to_latex(const Tensor<2, dim, ScalarType> &value)
      {
        std::string out = "\\begin{bmatrix}";
        for (unsigned int i = 0; i < dim; ++i)
          {
            out += get_separated_string_from(value[i].begin_raw(),
                                             value[i].end_raw(),
                                             "&");
            if (i < dim - 1)
              out += "\\\\";
          }
        out += "\\end{bmatrix}";
        return out;
      }
    };


    template <typename ScalarType>
    struct ConvertNumericToText<
      FullMatrix<ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      static std::string
      to_ascii(const FullMatrix<ScalarType> &value)
      {
        const std::size_t n_rows = value.m();
        const std::size_t n_cols = value.n();

        std::string out = "[";
        for (unsigned int i = 0; i < n_rows; ++i)
          {
            for (unsigned int j = 0; j < n_cols; ++j)
              {
                out += dealii::Utilities::to_string(value[i][j]);
                if (j < n_cols - 1)
                  out += ",";
              }
            if (i < n_rows - 1)
              out += ";";
          }
        out += "]";
        return out;
      }

      static std::string
      to_latex(const FullMatrix<ScalarType> &value)
      {
        const std::size_t n_rows = value.m();
        const std::size_t n_cols = value.n();

        std::string out = "\\begin{bmatrix}";
        for (unsigned int i = 0; i < n_rows; ++i)
          {
            for (unsigned int j = 0; j < n_cols; ++j)
              {
                out += dealii::Utilities::to_string(value[i][j]);
                if (j < n_cols - 1)
                  out += "&";
              }
            if (i < n_rows - 1)
              out += "\\\\";
          }
        out += "\\end{bmatrix}";
        return out;
      }
    };


    template <int dim, typename ScalarType>
    struct ConvertNumericToText<
      Tensor<3, dim, ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      static std::string
      to_ascii(const Tensor<3, dim, ScalarType> &value)
      {
        return ConvertNumericToText<FullMatrix<ScalarType>>::to_ascii(
          Physics::Notation::Kelvin::to_matrix(value));
      }

      static std::string
      to_latex(const Tensor<3, dim, ScalarType> &value)
      {
        return ConvertNumericToText<FullMatrix<ScalarType>>::to_latex(
          Physics::Notation::Kelvin::to_matrix(value));
      }
    };


    template <int dim, typename ScalarType>
    struct ConvertNumericToText<
      Tensor<4, dim, ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      static std::string
      to_ascii(const Tensor<4, dim, ScalarType> &value)
      {
        return ConvertNumericToText<FullMatrix<ScalarType>>::to_ascii(
          Physics::Notation::Kelvin::to_matrix(value));
      }

      static std::string
      to_latex(const Tensor<4, dim, ScalarType> &value)
      {
        return ConvertNumericToText<FullMatrix<ScalarType>>::to_latex(
          Physics::Notation::Kelvin::to_matrix(value));
      }
    };


    template <int dim, typename ScalarType>
    struct ConvertNumericToText<
      SymmetricTensor<2, dim, ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      // TODO: Is it worth copying this to a full matrix, instead of reproducing
      // the implementation here?

      static std::string
      to_ascii(const SymmetricTensor<2, dim, ScalarType> &value)
      {
        std::string out = "[";
        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j < dim; ++j)
              {
                out += dealii::Utilities::to_string(value[i][j]);
                if (j < dim - 1)
                  out += ",";
              }
            if (i < dim - 1)
              out += ";";
          }
        out += "]";
        return out;
      }

      static std::string
      to_latex(const SymmetricTensor<2, dim, ScalarType> &value)
      {
        std::string out = "\\begin{bmatrix}";
        for (unsigned int i = 0; i < dim; ++i)
          {
            for (unsigned int j = 0; j < dim; ++j)
              {
                out += dealii::Utilities::to_string(value[i][j]);
                if (j < dim - 1)
                  out += "&";
              }
            if (i < dim - 1)
              out += "\\\\";
          }
        out += "\\end{bmatrix}";
        return out;
      }
    };


    template <int dim, typename ScalarType>
    struct ConvertNumericToText<
      SymmetricTensor<4, dim, ScalarType>,
      typename std::enable_if<std::is_arithmetic<ScalarType>::value>::type>
    {
      static std::string
      to_ascii(const SymmetricTensor<4, dim, ScalarType> &value)
      {
        return ConvertNumericToText<FullMatrix<ScalarType>>::to_ascii(
          Physics::Notation::Kelvin::to_matrix(value));
      }

      static std::string
      to_latex(const SymmetricTensor<4, dim, ScalarType> &value)
      {
        return ConvertNumericToText<FullMatrix<ScalarType>>::to_latex(
          Physics::Notation::Kelvin::to_matrix(value));
      }
    };



    struct LaTeX
    {
      static constexpr char l_parenthesis[] = "\\left(";
      static constexpr char r_parenthesis[] = "\\right)";

      static constexpr char l_square_brace[] = "\\left[";
      static constexpr char r_square_brace[] = "\\right]";

      static constexpr char l_curly_brace[] = "\\left\\lbrace";
      static constexpr char r_curly_brace[] = "\\right\\rbrace";

      static constexpr char l_vert[] = "\\left\\vert";
      static constexpr char r_vert[] = "\\right\\vert";

      static constexpr char decrease_space[] = "\\!";

      static std::string
      decorate_latex_op(const std::string &op)
      {
        return "\\" + op;
      }

      static std::string
      decorate_jump(const std::string &op)
      {
        const std::string &lbrace = l_square_brace;
        const std::string &rbrace = r_square_brace;

        return lbrace + decrease_space + lbrace + op + rbrace + decrease_space +
               rbrace;
      }

      static std::string
      decorate_average(const std::string &op)
      {
        const std::string &lbrace = l_curly_brace;
        const std::string &rbrace = r_curly_brace;

        return lbrace + decrease_space + lbrace + op + rbrace + decrease_space +
               rbrace;
      }

      static std::string
      decorate_text(const std::string &text)
      {
        return "\\text{" + text + "}";
      }

      static std::string
      decorate_term(const std::string &term)
      {
        const std::string &lbrace = l_square_brace;
        const std::string &rbrace = r_square_brace;

        return lbrace + term + rbrace;
      }

      static std::string
      decorate_function_with_arguments(const std::string &function,
                                       const std::string &arguments)
      {
        const std::string &lbrace = l_parenthesis;
        const std::string &rbrace = r_parenthesis;

        return function + lbrace + arguments + rbrace;
      }

      static std::string
      decorate_tensor(const std::string &name, const unsigned int rank)
      {
        auto decorate = [&name](const std::string latex_cmd)
        { return latex_cmd + "{" + name + "}"; };

        switch (rank)
          {
            case (0):
              return decorate(""); // TODO[JPP]: return name;
              break;
            case (1):
              return decorate("\\mathrm");
              break;
            case (2):
              return decorate("\\mathbf");
              break;
            case (3):
              return decorate("\\mathfrak");
              break;
            case (4):
              return decorate("\\mathcal");
              break;
            default:
              break;
          }

        AssertThrow(false, ExcNotImplemented());
        return "";
      }

      static std::string
      decorate_fraction(const std::string &numerator,
                        const std::string &demoninator)
      {
        return "\\frac{" + numerator + "}{" + demoninator + "}";
      }

      static std::string
      decorate_superscript(const std::string &value,
                           const std::string &superscript)
      {
        return "{" + value + "}^{" + superscript + "}";
      }

      static std::string
      decorate_subscript(const std::string &value, const std::string &subscript)
      {
        return "{" + value + "}_{" + subscript + "}";
      }

      static std::string
      decorate_power(const std::string &base, const std::string &exponent)
      {
        return decorate_superscript(base, exponent);
      }

      static std::string
      decorate_integral(const std::string &integrand,
                        const std::string &infinitesimal_symbol,
                        const std::string &limits = "")
      {
        if (limits == "")
          return "\\int" + integrand + infinitesimal_symbol;
        else
          return "\\int\\limits_{" + limits + "}" + integrand +
                 infinitesimal_symbol;
      }

      static std::string
      decorate_differential(const std::string &symbol,
                            const unsigned int n_derivatives = 1,
                            const std::string &diff_symbol   = "\\mathrm{d}")
      {
        return get_symbol_diff(n_derivatives, diff_symbol) + symbol;
      }

      static std::string
      get_symbol_diff(const unsigned int n_derivatives = 1,
                      const std::string &diff_symbol   = "\\mathrm{d}")
      {
        Assert(
          n_derivatives > 0,
          ExcMessage(
            "Differential operation must take a positive number of derivatives."));
        if (n_derivatives == 1)
          return diff_symbol;
        else
          return diff_symbol + "^{" +
                 dealii::Utilities::to_string(n_derivatives) + "}";
      }

      static std::string
      get_symbol_multiply(const unsigned int n_contracting_indices)
      {
        switch (n_contracting_indices)
          {
            case (0):
              return "\\,";
              break;
            case (1):
              return " \\cdot ";
              break;
            case (2):
              return " \\colon ";
              break;
            case (3):
              return " \\vdots ";
              break;
            case (4):
              return " \\colon\\colon ";
              break;
            case (5):
              return " \\vdots\\colon ";
              break;
            case (6):
              return " \\vdots\\vdots ";
              break;
            default:
              break;
          }

        AssertThrow(false, ExcNotImplemented());
        return " * ";
      }

      static std::string
      get_symbol_outer_product(const unsigned int rank_lhs_op,
                               const unsigned int rank_rhs_op)
      {
        return (rank_lhs_op == 0 || rank_rhs_op == 0 ? " \\, " : " \\otimes ");
      }
    };

  } // namespace Utilities


  namespace internal
  {
    /**
     * Exception denoting that a class requires some specialization
     * in order to be used.
     */
    DeclExceptionMsg(ExcUnexpectedFunctionCall,
                     "This function should never be called, as it is "
                     "expected to be bypassed though the lack of availability "
                     "of a pointer at the calling site.");
  } // namespace internal

} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE


#endif // dealii_weakforms_utilities_h
