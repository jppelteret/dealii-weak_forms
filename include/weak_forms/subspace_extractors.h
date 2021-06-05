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
#include <weak_forms/types.h>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  namespace SubSpaceExtractors
  {
    struct Scalar
    {
      static const int rank = 0;

      Scalar(const types::field_index          field_index,
             const FEValuesExtractors::Scalar &extractor,
             const std::string &               field_ascii,
             const std::string &               field_latex)
        : extractor(extractor)
        , field_index(field_index)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      Scalar(const FEValuesExtractors::Scalar &extractor,
             const std::string &               field_ascii,
             const std::string &               field_latex)
        : Scalar(numbers::invalid_field_index,
                 extractor,
                 field_ascii,
                 field_latex)
      {}

      Scalar(const types::field_index field_index,
             const unsigned int       component,
             const std::string &      field_ascii,
             const std::string &      field_latex)
        : extractor(component)
        , field_index(field_index)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      Scalar(const unsigned int component,
             const std::string &field_ascii,
             const std::string &field_latex)
        : Scalar(numbers::invalid_field_index,
                 component,
                 field_ascii,
                 field_latex)
      {}

      const FEValuesExtractors::Scalar extractor;
      const types::field_index         field_index;
      const std::string                field_ascii;
      const std::string                field_latex;
    };

    struct Vector
    {
      static const int rank = 1;

      Vector(const types::field_index          field_index,
             const FEValuesExtractors::Vector &extractor,
             const std::string &               field_ascii,
             const std::string &               field_latex)
        : extractor(extractor)
        , field_index(field_index)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      Vector(const FEValuesExtractors::Vector &extractor,
             const std::string &               field_ascii,
             const std::string &               field_latex)
        : Vector(numbers::invalid_field_index,
                 extractor,
                 field_ascii,
                 field_latex)
      {}

      Vector(const types::field_index field_index,
             const unsigned int       first_component,
             const std::string &      field_ascii,
             const std::string &      field_latex)
        : extractor(first_component)
        , field_index(field_index)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      Vector(const unsigned int first_component,
             const std::string &field_ascii,
             const std::string &field_latex)
        : Vector(numbers::invalid_field_index,
                 first_component,
                 field_ascii,
                 field_latex)
      {}

      const FEValuesExtractors::Vector extractor;
      const types::field_index         field_index;
      const std::string                field_ascii;
      const std::string                field_latex;
    };

    template <int rank_>
    struct Tensor
    {
      static const int rank = rank_;

      Tensor(const types::field_index                field_index,
             const FEValuesExtractors::Tensor<rank> &extractor,
             const std::string &                     field_ascii,
             const std::string &                     field_latex)
        : extractor(extractor)
        , field_index(field_index)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      Tensor(const FEValuesExtractors::Tensor<rank> &extractor,
             const std::string &                     field_ascii,
             const std::string &                     field_latex)
        : Tensor(numbers::invalid_field_index,
                 extractor,
                 field_ascii,
                 field_latex)
      {}

      Tensor(const types::field_index field_index,
             const unsigned int       first_component,
             const std::string &      field_ascii,
             const std::string &      field_latex)
        : extractor(first_component)
        , field_index(field_index)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      Tensor(const unsigned int first_component,
             const std::string &field_ascii,
             const std::string &field_latex)
        : Tensor(numbers::invalid_field_index,
                 first_component,
                 field_ascii,
                 field_latex)
      {}

      const FEValuesExtractors::Tensor<rank> extractor;
      const types::field_index               field_index;
      const std::string                      field_ascii;
      const std::string                      field_latex;
    };

    template <int rank_>
    struct SymmetricTensor
    {
      static const int rank = rank_;

      SymmetricTensor(
        const types::field_index                         field_index,
        const FEValuesExtractors::SymmetricTensor<rank> &extractor,
        const std::string &                              field_ascii,
        const std::string &                              field_latex)
        : extractor(extractor)
        , field_index(field_index)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      SymmetricTensor(
        const FEValuesExtractors::SymmetricTensor<rank> &extractor,
        const std::string &                              field_ascii,
        const std::string &                              field_latex)
        : SymmetricTensor(numbers::invalid_field_index,
                          extractor,
                          field_ascii,
                          field_latex)
      {}

      SymmetricTensor(const types::field_index field_index,
                      const unsigned int       first_component,
                      const std::string &      field_ascii,
                      const std::string &      field_latex)
        : extractor(first_component)
        , field_index(field_index)
        , field_ascii(field_ascii)
        , field_latex(field_latex)
      {}

      SymmetricTensor(const unsigned int first_component,
                      const std::string &field_ascii,
                      const std::string &field_latex)
        : SymmetricTensor(numbers::invalid_field_index,
                          first_component,
                          field_ascii,
                          field_latex)
      {}

      const FEValuesExtractors::SymmetricTensor<rank> extractor;
      const types::field_index                        field_index;
      const std::string                               field_ascii;
      const std::string                               field_latex;
    };
  }; // namespace SubSpaceExtractors

} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE

#endif // dealii_weakforms_space_extractors_h
