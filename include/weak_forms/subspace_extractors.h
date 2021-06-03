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

#ifndef dealii_weakforms_space_extractors_h
#define dealii_weakforms_space_extractors_h

#include <deal.II/base/config.h>

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values_extractors.h>

#include <weak_forms/config.h>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  namespace SubSpaceExtractors
  {
    struct Scalar
    {
      static const int rank = 0;

      Scalar(const FEValuesExtractors::Scalar &extractor,
             const std::string &               field_ascii,
             const std::string &               field_latex)
        : extractor(extractor)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      Scalar(const unsigned int first_component,
             const std::string &field_ascii,
             const std::string &field_latex)
        : extractor(first_component)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      const FEValuesExtractors::Scalar extractor;
      const std::string                field_ascii;
      const std::string                field_latex;
    };

    struct Vector
    {
      static const int rank = 1;

      Vector(const FEValuesExtractors::Vector &extractor,
             const std::string &               field_ascii,
             const std::string &               field_latex)
        : extractor(extractor)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      Vector(const unsigned int first_component,
             const std::string &field_ascii,
             const std::string &field_latex)
        : extractor(first_component)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      const FEValuesExtractors::Vector extractor;
      const std::string                field_ascii;
      const std::string                field_latex;
    };

    template <int rank_>
    struct Tensor
    {
      static const int rank = rank_;

      Tensor(const FEValuesExtractors::Tensor<rank> &extractor,
             const std::string &                     field_ascii,
             const std::string &                     field_latex)
        : extractor(extractor)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      Tensor(const unsigned int first_component,
             const std::string &field_ascii,
             const std::string &field_latex)
        : extractor(first_component)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      const FEValuesExtractors::Tensor<rank> extractor;
      const std::string                      field_ascii;
      const std::string                      field_latex;
    };

    template <int rank_>
    struct SymmetricTensor
    {
      static const int rank = rank_;

      SymmetricTensor(
        const FEValuesExtractors::SymmetricTensor<rank> &extractor,
        const std::string &                              field_ascii,
        const std::string &                              field_latex)
        : extractor(extractor)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      SymmetricTensor(const unsigned int first_component,
                      const std::string &field_ascii,
                      const std::string &field_latex)
        : extractor(first_component)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      const FEValuesExtractors::SymmetricTensor<rank> extractor;
      const std::string                               field_ascii;
      const std::string                               field_latex;
    };
  }; // namespace SubSpaceExtractors

} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_space_extractors_h
