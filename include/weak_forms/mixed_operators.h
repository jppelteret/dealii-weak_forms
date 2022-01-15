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

#ifndef dealii_weakforms_mixed_operators_h
#define dealii_weakforms_mixed_operators_h

// Definitions where the operators are mixed with one another

#include <deal.II/base/config.h>

#include <deal.II/base/template_constraints.h>

#include <weak_forms/binary_operators.h>
#include <weak_forms/config.h>
#include <weak_forms/functors.h>
#include <weak_forms/symbolic_operators.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/unary_operators.h>

// Disambiguate operator* for symbolic integrals
#include <weak_forms/binary_integral_operators.h>
#include <weak_forms/symbolic_integral.h>
#include <weak_forms/unary_integral_operators.h>



WEAK_FORMS_NAMESPACE_OPEN



/* ================== Define unary operator overloads ================== */
/* ======================== Symbolic operators ========================= */
// See https://stackoverflow.com/a/12782697 for using multiple parameter packs


#define DEAL_II_UNARY_OP_OF_SYMBOLIC_OP(operator_name, unary_op_code)            \
  template <typename Op,                                                         \
            enum WeakForms::Operators::SymbolicOpCodes OpCode,                   \
            typename... OpArgs>                                                  \
  WeakForms::Operators::UnaryOp<                                                 \
    WeakForms::Operators::SymbolicOp<Op, OpCode, OpArgs...>,                     \
    WeakForms::Operators::UnaryOpCodes::unary_op_code>                           \
  operator_name(                                                                 \
    const WeakForms::Operators::SymbolicOp<Op, OpCode, OpArgs...> &operand)      \
  {                                                                              \
    using namespace WeakForms;                                                   \
    using namespace WeakForms::Operators;                                        \
                                                                                 \
    using SymbolicOpType = SymbolicOp<Op, OpCode, OpArgs...>;                    \
    using OpType         = UnaryOp<SymbolicOpType, UnaryOpCodes::unary_op_code>; \
                                                                                 \
    return OpType(operand);                                                      \
  }

// Scalar operations
DEAL_II_UNARY_OP_OF_SYMBOLIC_OP(operator-, negate)
DEAL_II_UNARY_OP_OF_SYMBOLIC_OP(sin, sine)
DEAL_II_UNARY_OP_OF_SYMBOLIC_OP(cos, cosine)
DEAL_II_UNARY_OP_OF_SYMBOLIC_OP(tan, tangent)
DEAL_II_UNARY_OP_OF_SYMBOLIC_OP(exp, exponential)
DEAL_II_UNARY_OP_OF_SYMBOLIC_OP(log, logarithm)
DEAL_II_UNARY_OP_OF_SYMBOLIC_OP(sqrt, square_root)
DEAL_II_UNARY_OP_OF_SYMBOLIC_OP(abs, absolute_value)

// Tensor operations
DEAL_II_UNARY_OP_OF_SYMBOLIC_OP(determinant, determinant)
DEAL_II_UNARY_OP_OF_SYMBOLIC_OP(invert, invert)
DEAL_II_UNARY_OP_OF_SYMBOLIC_OP(transpose, transpose)
DEAL_II_UNARY_OP_OF_SYMBOLIC_OP(symmetrize, symmetrize)

#undef DEAL_II_UNARY_OP_OF_SYMBOLIC_OP


/* ======================== Binary operators ========================= */


#define DEAL_II_UNARY_OP_OF_BINARY_OP(operator_name, unary_op_code)          \
  template <typename LhsOp,                                                  \
            typename RhsOp,                                                  \
            enum WeakForms::Operators::BinaryOpCodes OpCode>                 \
  WeakForms::Operators::UnaryOp<                                             \
    WeakForms::Operators::BinaryOp<LhsOp, RhsOp, OpCode>,                    \
    WeakForms::Operators::UnaryOpCodes::unary_op_code>                       \
  operator_name(                                                             \
    const WeakForms::Operators::BinaryOp<LhsOp, RhsOp, OpCode> &operand)     \
  {                                                                          \
    using namespace WeakForms;                                               \
    using namespace WeakForms::Operators;                                    \
                                                                             \
    using BinaryOpType = BinaryOp<LhsOp, RhsOp, OpCode>;                     \
    using OpType       = UnaryOp<BinaryOpType, UnaryOpCodes::unary_op_code>; \
                                                                             \
    return OpType(operand);                                                  \
  }

// Scalar operations
DEAL_II_UNARY_OP_OF_BINARY_OP(operator-, negate)
DEAL_II_UNARY_OP_OF_BINARY_OP(sin, sine)
DEAL_II_UNARY_OP_OF_BINARY_OP(cos, cosine)
DEAL_II_UNARY_OP_OF_BINARY_OP(tan, tangent)
DEAL_II_UNARY_OP_OF_BINARY_OP(exp, exponential)
DEAL_II_UNARY_OP_OF_BINARY_OP(log, logarithm)
DEAL_II_UNARY_OP_OF_BINARY_OP(sqrt, square_root)
DEAL_II_UNARY_OP_OF_BINARY_OP(abs, absolute_value)

// Tensor operations
DEAL_II_UNARY_OP_OF_BINARY_OP(determinant, determinant)
DEAL_II_UNARY_OP_OF_BINARY_OP(invert, invert)
DEAL_II_UNARY_OP_OF_BINARY_OP(transpose, transpose)
DEAL_II_UNARY_OP_OF_BINARY_OP(symmetrize, symmetrize)

#undef DEAL_II_UNARY_OP_OF_BINARY_OP



/* ================== Define binary operator overloads ================== */
/* ========================= Symbolic operators ========================= */

/**
 * Variant 1: LHS operand: Symbolic op ; RHS operand: Symbolic op
 * Variant 2: LHS operand: Symbolic op ; RHS operand: Binary op
 * Variant 3: LHS operand: Binary op ; RHS operand: Symbolic op
 */
#define DEAL_II_BINARY_OP_OF_SYMBOLIC_OP(operator_name, binary_op_code)      \
  template <typename LhsOp,                                                  \
            enum WeakForms::Operators::SymbolicOpCodes LhsOpCode,            \
            typename... LhsOpArgs,                                           \
            typename RhsOp,                                                  \
            enum WeakForms::Operators::SymbolicOpCodes RhsOpCode,            \
            typename... RhsOpArgs>                                           \
  WeakForms::Operators::BinaryOp<                                            \
    WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>,        \
    WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>,        \
    WeakForms::Operators::BinaryOpCodes::binary_op_code>                     \
  operator_name(                                                             \
    const WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>   \
      &lhs_op,                                                               \
    const WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>   \
      &rhs_op)                                                               \
  {                                                                          \
    using namespace WeakForms;                                               \
    using namespace WeakForms::Operators;                                    \
                                                                             \
    using LhsOpType = SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>;            \
    using RhsOpType = SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>;            \
    using OpType =                                                           \
      BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::binary_op_code>;         \
                                                                             \
    return OpType(lhs_op, rhs_op);                                           \
  }                                                                          \
                                                                             \
  template <typename LhsOp,                                                  \
            enum WeakForms::Operators::SymbolicOpCodes LhsOpCode,            \
            typename... LhsOpArgs,                                           \
            typename RhsOp1,                                                 \
            typename RhsOp2,                                                 \
            enum WeakForms::Operators::BinaryOpCodes RhsOpCode>              \
  WeakForms::Operators::BinaryOp<                                            \
    WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>,        \
    WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,               \
    WeakForms::Operators::BinaryOpCodes::binary_op_code>                     \
  operator_name(                                                             \
    const WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>   \
      &                                                              lhs_op, \
    const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op) \
  {                                                                          \
    using namespace WeakForms;                                               \
    using namespace WeakForms::Operators;                                    \
                                                                             \
    using LhsOpType = SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>;            \
    using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode>;                   \
    using OpType =                                                           \
      BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::binary_op_code>;         \
                                                                             \
    return OpType(lhs_op, rhs_op);                                           \
  }                                                                          \
                                                                             \
  template <typename LhsOp1,                                                 \
            typename LhsOp2,                                                 \
            enum WeakForms::Operators::BinaryOpCodes LhsOpCode,              \
            typename RhsOp,                                                  \
            enum WeakForms::Operators::SymbolicOpCodes RhsOpCode,            \
            typename... RhsOpArgs>                                           \
  WeakForms::Operators::BinaryOp<                                            \
    WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,               \
    WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>,        \
    WeakForms::Operators::BinaryOpCodes::binary_op_code>                     \
  operator_name(                                                             \
    const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op, \
    const WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>   \
      &rhs_op)                                                               \
  {                                                                          \
    using namespace WeakForms;                                               \
    using namespace WeakForms::Operators;                                    \
                                                                             \
    using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;                   \
    using RhsOpType = SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>;            \
    using OpType =                                                           \
      BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::binary_op_code>;         \
                                                                             \
    return OpType(lhs_op, rhs_op);                                           \
  }

// Arithmetic operations
DEAL_II_BINARY_OP_OF_SYMBOLIC_OP(operator+, add)
DEAL_II_BINARY_OP_OF_SYMBOLIC_OP(operator-, subtract)
DEAL_II_BINARY_OP_OF_SYMBOLIC_OP(operator*, multiply)
DEAL_II_BINARY_OP_OF_SYMBOLIC_OP(operator/, divide)

// Scalar operations
DEAL_II_BINARY_OP_OF_SYMBOLIC_OP(pow, power)
DEAL_II_BINARY_OP_OF_SYMBOLIC_OP(max, maximum)
DEAL_II_BINARY_OP_OF_SYMBOLIC_OP(min, minimum)

// Tensor operations
DEAL_II_BINARY_OP_OF_SYMBOLIC_OP(cross_product, cross_product)
DEAL_II_BINARY_OP_OF_SYMBOLIC_OP(schur_product, schur_product)
DEAL_II_BINARY_OP_OF_SYMBOLIC_OP(outer_product, outer_product)

// Tensor contractions
DEAL_II_BINARY_OP_OF_SYMBOLIC_OP(scalar_product, scalar_product)
DEAL_II_BINARY_OP_OF_SYMBOLIC_OP(double_contract,
                                 double_contract) // SymmetricTensor

#undef DEAL_II_BINARY_OP_OF_SYMBOLIC_OP


// Tensor contractions with extra template arguments

/**
 * Variant 1: LHS operand: Symbolic op ; RHS operand: Symbolic op
 * Variant 2: LHS operand: Symbolic op ; RHS operand: Binary op
 * Variant 3: LHS operand: Binary op ; RHS operand: Symbolic op
 */
#define DEAL_II_TENSOR_CONTRACTION_BINARY_OP_OF_SYMBOLIC_OP(operator_name,   \
                                                            binary_op_code)  \
  template <INDEX_PACK_TEMPLATE,                                             \
            typename LhsOp,                                                  \
            enum WeakForms::Operators::SymbolicOpCodes LhsOpCode,            \
            typename... LhsOpArgs,                                           \
            typename RhsOp,                                                  \
            enum WeakForms::Operators::SymbolicOpCodes RhsOpCode,            \
            typename... RhsOpArgs>                                           \
  WeakForms::Operators::BinaryOp<                                            \
    WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>,        \
    WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>,        \
    WeakForms::Operators::BinaryOpCodes::binary_op_code,                     \
    typename std::enable_if<                                                 \
      !WeakForms::is_integral_op<                                            \
        WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>>::  \
        value &&                                                             \
      !WeakForms::is_integral_op<                                            \
        WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>>::  \
        value>::type,                                                        \
    INDEX_PACK_EXPANDED>                                                     \
  operator_name(                                                             \
    const WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>   \
      &lhs_op,                                                               \
    const WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>   \
      &rhs_op)                                                               \
  {                                                                          \
    using namespace WeakForms;                                               \
    using namespace WeakForms::Operators;                                    \
                                                                             \
    using LhsOpType = SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>;            \
    using RhsOpType = SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>;            \
    using OpType    = BinaryOp<                                              \
      LhsOpType,                                                          \
      RhsOpType,                                                          \
      BinaryOpCodes::binary_op_code,                                      \
      typename std::enable_if<!is_integral_op<LhsOpType>::value &&        \
                              !is_integral_op<RhsOpType>::value>::type,   \
      INDEX_PACK_EXPANDED>;                                               \
                                                                             \
    return OpType(lhs_op, rhs_op);                                           \
  }                                                                          \
                                                                             \
  template <INDEX_PACK_TEMPLATE,                                             \
            typename LhsOp,                                                  \
            enum WeakForms::Operators::SymbolicOpCodes LhsOpCode,            \
            typename... LhsOpArgs,                                           \
            typename RhsOp1,                                                 \
            typename RhsOp2,                                                 \
            enum WeakForms::Operators::BinaryOpCodes RhsOpCode>              \
  WeakForms::Operators::BinaryOp<                                            \
    WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>,        \
    WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,               \
    WeakForms::Operators::BinaryOpCodes::binary_op_code,                     \
    typename std::enable_if<                                                 \
      !WeakForms::is_integral_op<                                            \
        WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>>::  \
        value &&                                                             \
      !WeakForms::is_integral_op<                                            \
        WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>>::value>:: \
      type,                                                                  \
    INDEX_PACK_EXPANDED>                                                     \
  operator_name(                                                             \
    const WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>   \
      &                                                              lhs_op, \
    const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op) \
  {                                                                          \
    using namespace WeakForms;                                               \
    using namespace WeakForms::Operators;                                    \
                                                                             \
    using LhsOpType = SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>;            \
    using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode>;                   \
    using OpType    = BinaryOp<                                              \
      LhsOpType,                                                          \
      RhsOpType,                                                          \
      BinaryOpCodes::binary_op_code,                                      \
      typename std::enable_if<!is_integral_op<LhsOpType>::value &&        \
                              !is_integral_op<RhsOpType>::value>::type,   \
      INDEX_PACK_EXPANDED>;                                               \
                                                                             \
    return OpType(lhs_op, rhs_op);                                           \
  }                                                                          \
                                                                             \
  template <INDEX_PACK_TEMPLATE,                                             \
            typename LhsOp1,                                                 \
            typename LhsOp2,                                                 \
            enum WeakForms::Operators::BinaryOpCodes LhsOpCode,              \
            typename RhsOp,                                                  \
            enum WeakForms::Operators::SymbolicOpCodes RhsOpCode,            \
            typename... RhsOpArgs>                                           \
  WeakForms::Operators::BinaryOp<                                            \
    WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,               \
    WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>,        \
    WeakForms::Operators::BinaryOpCodes::binary_op_code,                     \
    typename std::enable_if<                                                 \
      !WeakForms::is_integral_op<                                            \
        WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>>::value && \
      !WeakForms::is_integral_op<                                            \
        WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>>::  \
        value>::type,                                                        \
    INDEX_PACK_EXPANDED>                                                     \
  operator_name(                                                             \
    const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op, \
    const WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>   \
      &rhs_op)                                                               \
  {                                                                          \
    using namespace WeakForms;                                               \
    using namespace WeakForms::Operators;                                    \
                                                                             \
    using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;                   \
    using RhsOpType = SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>;            \
    using OpType    = BinaryOp<                                              \
      LhsOpType,                                                          \
      RhsOpType,                                                          \
      BinaryOpCodes::binary_op_code,                                      \
      typename std::enable_if<!is_integral_op<LhsOpType>::value &&        \
                              !is_integral_op<RhsOpType>::value>::type,   \
      INDEX_PACK_EXPANDED>;                                               \
                                                                             \
    return OpType(lhs_op, rhs_op);                                           \
  }

// https://stackoverflow.com/questions/44268316/passing-a-template-type-into-a-macro
#define COMMA ,
#define INDEX_PACK_TEMPLATE int lhs_index COMMA int rhs_index
#define INDEX_PACK_EXPANDED \
  WeakForms::Operators::internal::TwoIndexPack<lhs_index COMMA rhs_index>
DEAL_II_TENSOR_CONTRACTION_BINARY_OP_OF_SYMBOLIC_OP(contract, contract)
#undef INDEX_PACK_EXPANDED
#undef INDEX_PACK_TEMPLATE

#define INDEX_PACK_TEMPLATE                                   \
  int lhs_index_1 COMMA int rhs_index_1 COMMA int lhs_index_2 \
    COMMA int rhs_index_2
#define INDEX_PACK_EXPANDED                      \
  WeakForms::Operators::internal::FourIndexPack< \
    lhs_index_1 COMMA rhs_index_1 COMMA lhs_index_2 COMMA rhs_index_2>
DEAL_II_TENSOR_CONTRACTION_BINARY_OP_OF_SYMBOLIC_OP(double_contract,
                                                    double_contract)
#undef INDEX_PACK_EXPANDED
#undef INDEX_PACK_TEMPLATE
#undef COMMA

#undef DEAL_II_TENSOR_CONTRACTION_BINARY_OP_OF_SYMBOLIC_OP


/* ========================= Unary operators ========================= */


/**
 * Variant 1: LHS operand: Unary op ; RHS operand: Unary op
 * Variant 2: LHS operand: Unary op ; RHS operand: Binary op
 * Variant 3: LHS operand: Binary op ; RHS operand: Unary op
 */
#define DEAL_II_BINARY_OP_OF_UNARY_OP(operator_name, binary_op_code)           \
  template <typename LhsOp,                                                    \
            enum WeakForms::Operators::UnaryOpCodes LhsOpCode,                 \
            typename RhsOp,                                                    \
            enum WeakForms::Operators::UnaryOpCodes RhsOpCode>                 \
  WeakForms::Operators::BinaryOp<                                              \
    WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode>,                           \
    WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode>,                           \
    WeakForms::Operators::BinaryOpCodes::binary_op_code>                       \
  operator_name(const WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode> &lhs_op, \
                const WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode> &rhs_op) \
  {                                                                            \
    using namespace WeakForms;                                                 \
    using namespace WeakForms::Operators;                                      \
                                                                               \
    using LhsOpType = UnaryOp<LhsOp, LhsOpCode>;                               \
    using RhsOpType = UnaryOp<RhsOp, RhsOpCode>;                               \
    using OpType =                                                             \
      BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::binary_op_code>;           \
                                                                               \
    return OpType(lhs_op, rhs_op);                                             \
  }                                                                            \
                                                                               \
  template <typename LhsOp,                                                    \
            enum WeakForms::Operators::UnaryOpCodes LhsOpCode,                 \
            typename RhsOp1,                                                   \
            typename RhsOp2,                                                   \
            enum WeakForms::Operators::BinaryOpCodes RhsOpCode>                \
  WeakForms::Operators::BinaryOp<                                              \
    WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode>,                           \
    WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,                 \
    WeakForms::Operators::BinaryOpCodes::binary_op_code>                       \
  operator_name(                                                               \
    const WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode> &          lhs_op,   \
    const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op)   \
  {                                                                            \
    using namespace WeakForms;                                                 \
    using namespace WeakForms::Operators;                                      \
                                                                               \
    using LhsOpType = UnaryOp<LhsOp, LhsOpCode>;                               \
    using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode>;                     \
    using OpType =                                                             \
      BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::binary_op_code>;           \
                                                                               \
    return OpType(lhs_op, rhs_op);                                             \
  }                                                                            \
                                                                               \
  template <typename LhsOp1,                                                   \
            typename LhsOp2,                                                   \
            enum WeakForms::Operators::BinaryOpCodes LhsOpCode,                \
            typename RhsOp,                                                    \
            enum WeakForms::Operators::UnaryOpCodes RhsOpCode>                 \
  WeakForms::Operators::BinaryOp<                                              \
    WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,                 \
    WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode>,                           \
    WeakForms::Operators::BinaryOpCodes::binary_op_code>                       \
  operator_name(                                                               \
    const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op,   \
    const WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode> &          rhs_op)   \
  {                                                                            \
    using namespace WeakForms;                                                 \
    using namespace WeakForms::Operators;                                      \
                                                                               \
    using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;                     \
    using RhsOpType = UnaryOp<RhsOp, RhsOpCode>;                               \
    using OpType =                                                             \
      BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::binary_op_code>;           \
                                                                               \
    return OpType(lhs_op, rhs_op);                                             \
  }

// Arithmetic operations
DEAL_II_BINARY_OP_OF_UNARY_OP(operator+, add)
DEAL_II_BINARY_OP_OF_UNARY_OP(operator-, subtract)
DEAL_II_BINARY_OP_OF_UNARY_OP(operator*, multiply)
DEAL_II_BINARY_OP_OF_UNARY_OP(operator/, divide)

// Scalar operations
DEAL_II_BINARY_OP_OF_UNARY_OP(pow, power)
DEAL_II_BINARY_OP_OF_UNARY_OP(max, maximum)
DEAL_II_BINARY_OP_OF_UNARY_OP(min, minimum)

// Tensor operations
DEAL_II_BINARY_OP_OF_UNARY_OP(cross_product, cross_product)
DEAL_II_BINARY_OP_OF_UNARY_OP(schur_product, schur_product)
DEAL_II_BINARY_OP_OF_UNARY_OP(outer_product, outer_product)

// Tensor contractions
DEAL_II_BINARY_OP_OF_UNARY_OP(scalar_product, scalar_product)
DEAL_II_BINARY_OP_OF_UNARY_OP(double_contract,
                              double_contract) // SymmetricTensor

#undef DEAL_II_BINARY_OP_OF_UNARY_OP


// Tensor contractions with extra template arguments

/**
 * Variant 1: LHS operand: Unary op ; RHS operand: Unary op
 * Variant 2: LHS operand: Unary op ; RHS operand: Binary op
 * Variant 3: LHS operand: Binary op ; RHS operand: Unary op
 */
#define DEAL_II_TENSOR_CONTRACTION_BINARY_OP_OF_UNARY_OP(operator_name,        \
                                                         binary_op_code)       \
  template <INDEX_PACK_TEMPLATE,                                               \
            typename LhsOp,                                                    \
            enum WeakForms::Operators::UnaryOpCodes LhsOpCode,                 \
            typename RhsOp,                                                    \
            enum WeakForms::Operators::UnaryOpCodes RhsOpCode>                 \
  WeakForms::Operators::BinaryOp<                                              \
    WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode>,                           \
    WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode>,                           \
    WeakForms::Operators::BinaryOpCodes::binary_op_code,                       \
    typename std::enable_if<                                                   \
      !WeakForms::is_integral_op<                                              \
        WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode>>::value &&             \
      !WeakForms::is_integral_op<                                              \
        WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode>>::value>::type,        \
    INDEX_PACK_EXPANDED>                                                       \
  operator_name(const WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode> &lhs_op, \
                const WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode> &rhs_op) \
  {                                                                            \
    using namespace WeakForms;                                                 \
    using namespace WeakForms::Operators;                                      \
                                                                               \
    using LhsOpType = UnaryOp<LhsOp, LhsOpCode>;                               \
    using RhsOpType = UnaryOp<RhsOp, RhsOpCode>;                               \
    using OpType    = BinaryOp<                                                \
      LhsOpType,                                                            \
      RhsOpType,                                                            \
      BinaryOpCodes::binary_op_code,                                        \
      typename std::enable_if<!is_integral_op<LhsOpType>::value &&          \
                              !is_integral_op<RhsOpType>::value>::type,     \
      INDEX_PACK_EXPANDED>;                                                 \
                                                                               \
    return OpType(lhs_op, rhs_op);                                             \
  }                                                                            \
                                                                               \
  template <INDEX_PACK_TEMPLATE,                                               \
            typename LhsOp,                                                    \
            enum WeakForms::Operators::UnaryOpCodes LhsOpCode,                 \
            typename RhsOp1,                                                   \
            typename RhsOp2,                                                   \
            enum WeakForms::Operators::BinaryOpCodes RhsOpCode>                \
  WeakForms::Operators::BinaryOp<                                              \
    WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode>,                           \
    WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>,                 \
    WeakForms::Operators::BinaryOpCodes::binary_op_code,                       \
    typename std::enable_if<                                                   \
      !WeakForms::is_integral_op<                                              \
        WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode>>::value &&             \
      !WeakForms::is_integral_op<                                              \
        WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode>>::value>::   \
      type,                                                                    \
    INDEX_PACK_EXPANDED>                                                       \
  operator_name(                                                               \
    const WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode> &          lhs_op,   \
    const WeakForms::Operators::BinaryOp<RhsOp1, RhsOp2, RhsOpCode> &rhs_op)   \
  {                                                                            \
    using namespace WeakForms;                                                 \
    using namespace WeakForms::Operators;                                      \
                                                                               \
    using LhsOpType = UnaryOp<LhsOp, LhsOpCode>;                               \
    using RhsOpType = BinaryOp<RhsOp1, RhsOp2, RhsOpCode>;                     \
    using OpType    = BinaryOp<                                                \
      LhsOpType,                                                            \
      RhsOpType,                                                            \
      BinaryOpCodes::binary_op_code,                                        \
      typename std::enable_if<!is_integral_op<LhsOpType>::value &&          \
                              !is_integral_op<RhsOpType>::value>::type,     \
      INDEX_PACK_EXPANDED>;                                                 \
                                                                               \
    return OpType(lhs_op, rhs_op);                                             \
  }                                                                            \
                                                                               \
  template <INDEX_PACK_TEMPLATE,                                               \
            typename LhsOp1,                                                   \
            typename LhsOp2,                                                   \
            enum WeakForms::Operators::BinaryOpCodes LhsOpCode,                \
            typename RhsOp,                                                    \
            enum WeakForms::Operators::UnaryOpCodes RhsOpCode>                 \
  WeakForms::Operators::BinaryOp<                                              \
    WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>,                 \
    WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode>,                           \
    WeakForms::Operators::BinaryOpCodes::binary_op_code,                       \
    typename std::enable_if<                                                   \
      !WeakForms::is_integral_op<                                              \
        WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode>>::value &&   \
      !WeakForms::is_integral_op<                                              \
        WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode>>::value>::type,        \
    INDEX_PACK_EXPANDED>                                                       \
  operator_name(                                                               \
    const WeakForms::Operators::BinaryOp<LhsOp1, LhsOp2, LhsOpCode> &lhs_op,   \
    const WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode> &          rhs_op)   \
  {                                                                            \
    using namespace WeakForms;                                                 \
    using namespace WeakForms::Operators;                                      \
                                                                               \
    using LhsOpType = BinaryOp<LhsOp1, LhsOp2, LhsOpCode>;                     \
    using RhsOpType = UnaryOp<RhsOp, RhsOpCode>;                               \
    using OpType    = BinaryOp<                                                \
      LhsOpType,                                                            \
      RhsOpType,                                                            \
      BinaryOpCodes::binary_op_code,                                        \
      typename std::enable_if<!is_integral_op<LhsOpType>::value &&          \
                              !is_integral_op<RhsOpType>::value>::type,     \
      INDEX_PACK_EXPANDED>;                                                 \
                                                                               \
    return OpType(lhs_op, rhs_op);                                             \
  }

// https://stackoverflow.com/questions/44268316/passing-a-template-type-into-a-macro
#define COMMA ,
#define INDEX_PACK_TEMPLATE int lhs_index COMMA int rhs_index
#define INDEX_PACK_EXPANDED \
  WeakForms::Operators::internal::TwoIndexPack<lhs_index COMMA rhs_index>
DEAL_II_TENSOR_CONTRACTION_BINARY_OP_OF_UNARY_OP(contract, contract)
#undef INDEX_PACK_EXPANDED
#undef INDEX_PACK_TEMPLATE

#define INDEX_PACK_TEMPLATE                                   \
  int lhs_index_1 COMMA int rhs_index_1 COMMA int lhs_index_2 \
    COMMA int rhs_index_2
#define INDEX_PACK_EXPANDED                      \
  WeakForms::Operators::internal::FourIndexPack< \
    lhs_index_1 COMMA rhs_index_1 COMMA lhs_index_2 COMMA rhs_index_2>
DEAL_II_TENSOR_CONTRACTION_BINARY_OP_OF_UNARY_OP(double_contract,
                                                 double_contract)
#undef INDEX_PACK_EXPANDED
#undef INDEX_PACK_TEMPLATE
#undef COMMA

#undef DEAL_II_TENSOR_CONTRACTION_BINARY_OP_OF_UNARY_OP


/* ==================== Symbolic and Unary operators ==================== */

/**
 * Variant 1: LHS operand: Symbolic op ; RHS operand: Unary op
 * Variant 2: LHS operand: Unary op ; RHS operand: Symbolic op
 */
#define DEAL_II_BINARY_OP_OF_SYMBOLIC_AND_UNARY_OP(operator_name,          \
                                                   binary_op_code)         \
  template <typename LhsOp,                                                \
            enum WeakForms::Operators::SymbolicOpCodes LhsOpCode,          \
            typename... LhsOpArgs,                                         \
            typename RhsOp,                                                \
            enum WeakForms::Operators::UnaryOpCodes RhsOpCode>             \
  WeakForms::Operators::BinaryOp<                                          \
    WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>,      \
    WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode>,                       \
    WeakForms::Operators::BinaryOpCodes::binary_op_code>                   \
  operator_name(                                                           \
    const WeakForms::Operators::SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...> \
      &                                                    lhs_op,         \
    const WeakForms::Operators::UnaryOp<RhsOp, RhsOpCode> &rhs_op)         \
  {                                                                        \
    using namespace WeakForms;                                             \
    using namespace WeakForms::Operators;                                  \
                                                                           \
    using LhsOpType = SymbolicOp<LhsOp, LhsOpCode, LhsOpArgs...>;          \
    using RhsOpType = UnaryOp<RhsOp, RhsOpCode>;                           \
    using OpType =                                                         \
      BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::binary_op_code>;       \
                                                                           \
    return OpType(lhs_op, rhs_op);                                         \
  }                                                                        \
                                                                           \
  template <typename LhsOp,                                                \
            enum WeakForms::Operators::UnaryOpCodes LhsOpCode,             \
            typename RhsOp,                                                \
            enum WeakForms::Operators::SymbolicOpCodes RhsOpCode,          \
            typename... RhsOpArgs>                                         \
  WeakForms::Operators::BinaryOp<                                          \
    WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode>,                       \
    WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>,      \
    WeakForms::Operators::BinaryOpCodes::binary_op_code>                   \
  operator_name(                                                           \
    const WeakForms::Operators::UnaryOp<LhsOp, LhsOpCode> &lhs_op,         \
    const WeakForms::Operators::SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...> \
      &rhs_op)                                                             \
  {                                                                        \
    using namespace WeakForms;                                             \
    using namespace WeakForms::Operators;                                  \
                                                                           \
    using LhsOpType = UnaryOp<LhsOp, LhsOpCode>;                           \
    using RhsOpType = SymbolicOp<RhsOp, RhsOpCode, RhsOpArgs...>;          \
    using OpType =                                                         \
      BinaryOp<LhsOpType, RhsOpType, BinaryOpCodes::binary_op_code>;       \
                                                                           \
    return OpType(lhs_op, rhs_op);                                         \
  }

DEAL_II_BINARY_OP_OF_SYMBOLIC_AND_UNARY_OP(operator+, add)
DEAL_II_BINARY_OP_OF_SYMBOLIC_AND_UNARY_OP(operator-, subtract)
DEAL_II_BINARY_OP_OF_SYMBOLIC_AND_UNARY_OP(operator*, multiply)
DEAL_II_BINARY_OP_OF_SYMBOLIC_AND_UNARY_OP(operator/, divide)
DEAL_II_BINARY_OP_OF_SYMBOLIC_AND_UNARY_OP(pow, power)

#undef DEAL_II_BINARY_OP_OF_SYMBOLIC_AND_UNARY_OP


// ============================= Addition =============================


// Scalar + Symbolic Operator
template <
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    std::is_same<ScalarType,
                 typename EnableIfScalar<ScalarType>::type>::value &&
    SymbolicOp::rank == 0 &&
    WeakForms::is_compatible_with_scalar_arithmetic<SymbolicOp>::value>::type>
auto
operator+(const ScalarType &value, const SymbolicOp &op)
{
  constexpr int dim      = SymbolicOp::dimension;
  constexpr int spacedim = SymbolicOp::space_dimension;
  return WeakForms::constant_scalar<dim, spacedim>(value) + op;
}


// Symbolic Operator + Scalar
template <
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    std::is_same<ScalarType,
                 typename EnableIfScalar<ScalarType>::type>::value &&
    SymbolicOp::rank == 0 &&
    WeakForms::is_compatible_with_scalar_arithmetic<SymbolicOp>::value>::type>
auto
operator+(const SymbolicOp &op, const ScalarType &value)
{
  constexpr int dim      = SymbolicOp::dimension;
  constexpr int spacedim = SymbolicOp::space_dimension;
  return op + WeakForms::constant_scalar<dim, spacedim>(value);
}


// Tensor + Symbolic Operator
template <
  int rank,
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    SymbolicOp::rank == rank &&
    WeakForms::is_compatible_with_tensor_arithmetic<SymbolicOp>::value>::type>
auto
operator+(const Tensor<rank, spacedim, ScalarType> &value, const SymbolicOp &op)
{
  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return WeakForms::constant_tensor<rank, spacedim>(value) + op;
}


// Symbolic Operator + Tensor
template <
  int rank,
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    SymbolicOp::rank == rank &&
    WeakForms::is_compatible_with_tensor_arithmetic<SymbolicOp>::value>::type>
auto
operator+(const SymbolicOp &op, const Tensor<rank, spacedim, ScalarType> &value)
{
  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return op + WeakForms::constant_tensor<rank, spacedim>(value);
}


// SymmetricTensor + Symbolic Operator
template <
  int rank,
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    SymbolicOp::rank == rank &&
    WeakForms::is_compatible_with_tensor_arithmetic<SymbolicOp>::value>::type>
auto
operator+(const SymmetricTensor<rank, spacedim, ScalarType> &value,
          const SymbolicOp &                                 op)
{
  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return WeakForms::constant_symmetric_tensor<rank, spacedim>(value) + op;
}


// Symbolic Operator + SymmetricTensor
template <
  int rank,
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    SymbolicOp::rank == rank &&
    WeakForms::is_compatible_with_tensor_arithmetic<SymbolicOp>::value>::type>
auto
operator+(const SymbolicOp &                                 op,
          const SymmetricTensor<rank, spacedim, ScalarType> &value)
{
  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return op + WeakForms::constant_symmetric_tensor<rank, spacedim>(value);
}


// ============================= Subtraction =============================


// Scalar - Symbolic Operator
template <
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    std::is_same<ScalarType,
                 typename EnableIfScalar<ScalarType>::type>::value &&
    SymbolicOp::rank == 0 &&
    WeakForms::is_compatible_with_scalar_arithmetic<SymbolicOp>::value>::type>
auto
operator-(const ScalarType &value, const SymbolicOp &op)
{
  constexpr int dim      = SymbolicOp::dimension;
  constexpr int spacedim = SymbolicOp::space_dimension;
  return WeakForms::constant_scalar<dim, spacedim>(value) - op;
}


// Symbolic Operator - Scalar
template <
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    std::is_same<ScalarType,
                 typename EnableIfScalar<ScalarType>::type>::value &&
    SymbolicOp::rank == 0 &&
    WeakForms::is_compatible_with_scalar_arithmetic<SymbolicOp>::value>::type>
auto
operator-(const SymbolicOp &op, const ScalarType &value)
{
  constexpr int dim      = SymbolicOp::dimension;
  constexpr int spacedim = SymbolicOp::space_dimension;
  return op - WeakForms::constant_scalar<dim, spacedim>(value);
}


// Tensor - Symbolic Operator
template <
  int rank,
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    SymbolicOp::rank == rank &&
    WeakForms::is_compatible_with_tensor_arithmetic<SymbolicOp>::value>::type>
auto
operator-(const Tensor<rank, spacedim, ScalarType> &value, const SymbolicOp &op)
{
  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return WeakForms::constant_tensor<rank, spacedim>(value) - op;
}


// Symbolic Operator - Tensor
template <
  int rank,
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    SymbolicOp::rank == rank &&
    WeakForms::is_compatible_with_tensor_arithmetic<SymbolicOp>::value>::type>
auto
operator-(const SymbolicOp &op, const Tensor<rank, spacedim, ScalarType> &value)
{
  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return op - WeakForms::constant_tensor<rank, spacedim>(value);
}


// SymmetricTensor - Symbolic Operator
template <
  int rank,
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    SymbolicOp::rank == rank &&
    WeakForms::is_compatible_with_tensor_arithmetic<SymbolicOp>::value>::type>
auto
operator-(const SymmetricTensor<rank, spacedim, ScalarType> &value,
          const SymbolicOp &                                 op)
{
  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return WeakForms::constant_symmetric_tensor<rank, spacedim>(value) - op;
}


// Symbolic Operator - SymmetricTensor
template <
  int rank,
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    SymbolicOp::rank == rank &&
    WeakForms::is_compatible_with_tensor_arithmetic<SymbolicOp>::value>::type>
auto
operator-(const SymbolicOp &                                 op,
          const SymmetricTensor<rank, spacedim, ScalarType> &value)
{
  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return op - WeakForms::constant_symmetric_tensor<rank, spacedim>(value);
}


// ============================= Multiplication =============================


// Scalar * Symbolic Operator
template <
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    std::is_same<ScalarType,
                 typename EnableIfScalar<ScalarType>::type>::value &&
    WeakForms::is_compatible_with_scalar_arithmetic<SymbolicOp>::value>::type>
auto
operator*(const ScalarType &value, const SymbolicOp &op)
{
  static_assert(!WeakForms::is_integral_op<SymbolicOp>::value,
                "Expected not to be an integral op");

  constexpr int dim      = SymbolicOp::dimension;
  constexpr int spacedim = SymbolicOp::space_dimension;
  return WeakForms::constant_scalar<dim, spacedim>(value) * op;
}


// Symbolic Operator * Scalar
template <
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    std::is_same<ScalarType,
                 typename EnableIfScalar<ScalarType>::type>::value &&
    WeakForms::is_compatible_with_scalar_arithmetic<SymbolicOp>::value>::type>
auto
operator*(const SymbolicOp &op, const ScalarType &value)
{
  static_assert(!WeakForms::is_integral_op<SymbolicOp>::value,
                "Expected not to be an integral op");

  constexpr int dim      = SymbolicOp::dimension;
  constexpr int spacedim = SymbolicOp::space_dimension;
  return op * WeakForms::constant_scalar<dim, spacedim>(value);
}


// Tensor * Symbolic Operator
template <
  int rank,
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    WeakForms::is_compatible_with_tensor_arithmetic<SymbolicOp>::value>::type>
auto
operator*(const Tensor<rank, spacedim, ScalarType> &value, const SymbolicOp &op)
{
  static_assert(!WeakForms::is_integral_op<SymbolicOp>::value,
                "Expected not to be an integral op");

  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return WeakForms::constant_tensor<rank, spacedim>(value) * op;
}


// Symbolic Operator * Tensor
template <
  int rank,
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    WeakForms::is_compatible_with_tensor_arithmetic<SymbolicOp>::value>::type>
auto
operator*(const SymbolicOp &op, const Tensor<rank, spacedim, ScalarType> &value)
{
  static_assert(!WeakForms::is_integral_op<SymbolicOp>::value,
                "Expected not to be an integral op");

  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return op * WeakForms::constant_tensor<rank, spacedim>(value);
}


// SymmetricTensor * Symbolic Operator
template <
  int rank,
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    WeakForms::is_compatible_with_tensor_arithmetic<SymbolicOp>::value>::type>
auto
operator*(const SymmetricTensor<rank, spacedim, ScalarType> &value,
          const SymbolicOp &                                 op)
{
  static_assert(!WeakForms::is_integral_op<SymbolicOp>::value,
                "Expected not to be an integral op");

  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return WeakForms::constant_symmetric_tensor<rank, spacedim>(value) * op;
}


// Symbolic Operator * SymmetricTensor
template <
  int rank,
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    WeakForms::is_compatible_with_tensor_arithmetic<SymbolicOp>::value>::type>
auto
operator*(const SymbolicOp &                                 op,
          const SymmetricTensor<rank, spacedim, ScalarType> &value)
{
  static_assert(!WeakForms::is_integral_op<SymbolicOp>::value,
                "Expected not to be an integral op");

  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return op * WeakForms::constant_symmetric_tensor<rank, spacedim>(value);
}


// ============================= Division =============================


// Symbolic Operator / Scalar
template <
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    std::is_same<ScalarType,
                 typename EnableIfScalar<ScalarType>::type>::value &&
    WeakForms::is_compatible_with_scalar_arithmetic<SymbolicOp>::value>::type>
auto
operator/(const SymbolicOp &op, const ScalarType &value)
{
  static_assert(!WeakForms::is_integral_op<SymbolicOp>::value,
                "Expected not to be an integral op");

  constexpr int dim      = SymbolicOp::dimension;
  constexpr int spacedim = SymbolicOp::space_dimension;
  return op / WeakForms::constant_scalar<dim, spacedim>(value);
}


// Scalar / Symbolic Operator
template <
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    std::is_same<ScalarType,
                 typename EnableIfScalar<ScalarType>::type>::value &&
    WeakForms::is_compatible_with_scalar_arithmetic<SymbolicOp>::value>::type>
auto
operator/(const ScalarType &value, const SymbolicOp &op)
{
  static_assert(!WeakForms::is_integral_op<SymbolicOp>::value,
                "Expected not to be an integral op");

  constexpr int dim      = SymbolicOp::dimension;
  constexpr int spacedim = SymbolicOp::space_dimension;
  return WeakForms::constant_scalar<dim, spacedim>(value) / op;
}


// Symbolic Operator / Tensor (rank-0)
template <
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    WeakForms::is_compatible_with_tensor_arithmetic<SymbolicOp>::value>::type>
auto
operator/(const SymbolicOp &op, const Tensor<0, spacedim, ScalarType> &value)
{
  static_assert(!WeakForms::is_integral_op<SymbolicOp>::value,
                "Expected not to be an integral op");

  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return op / WeakForms::constant_tensor<0, spacedim>(value);
}


// Tensor / Symbolic Operator
template <
  int rank,
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    WeakForms::is_compatible_with_tensor_arithmetic<SymbolicOp>::value>::type>
auto
operator/(const Tensor<rank, spacedim, ScalarType> &value, const SymbolicOp &op)
{
  static_assert(!WeakForms::is_integral_op<SymbolicOp>::value,
                "Expected not to be an integral op");

  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return WeakForms::constant_tensor<rank, spacedim>(value) / op;
}


// SymmetricTensor / Symbolic Operator
template <
  int rank,
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    WeakForms::is_compatible_with_tensor_arithmetic<SymbolicOp>::value>::type>
auto
operator/(const SymmetricTensor<rank, spacedim, ScalarType> &value,
          const SymbolicOp &                                 op)
{
  static_assert(!WeakForms::is_integral_op<SymbolicOp>::value,
                "Expected not to be an integral op");

  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return WeakForms::constant_symmetric_tensor<rank, spacedim>(value) / op;
}


// ============================= Power =============================


// Symbolic Operator ^ Scalar
template <
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    std::is_same<ScalarType,
                 typename EnableIfScalar<ScalarType>::type>::value &&
    WeakForms::is_compatible_with_scalar_arithmetic<SymbolicOp>::value>::type>
auto
pow(const SymbolicOp &op, const ScalarType &value)
{
  constexpr int dim      = SymbolicOp::dimension;
  constexpr int spacedim = SymbolicOp::space_dimension;
  return pow(op, WeakForms::constant_scalar<dim, spacedim>(value));
}


// Scalar ^ Symbolic Operator
template <
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    std::is_same<ScalarType,
                 typename EnableIfScalar<ScalarType>::type>::value &&
    WeakForms::is_compatible_with_scalar_arithmetic<SymbolicOp>::value>::type>
auto
pow(const ScalarType &value, const SymbolicOp &op)
{
  constexpr int dim      = SymbolicOp::dimension;
  constexpr int spacedim = SymbolicOp::space_dimension;
  return pow(WeakForms::constant_scalar<dim, spacedim>(value), op);
}


// Symbolic Operator ^ Tensor (rank-0)
template <
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    WeakForms::is_compatible_with_scalar_arithmetic<SymbolicOp>::value>::type>
auto
pow(const SymbolicOp &op, const Tensor<0, spacedim, ScalarType> &value)
{
  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return pow(op, WeakForms::constant_tensor<0, spacedim>(value));
}


// Tensor ^ Symbolic Operator
template <
  int spacedim,
  typename ScalarType,
  typename SymbolicOp,
  typename = typename std::enable_if<
    WeakForms::is_compatible_with_scalar_arithmetic<SymbolicOp>::value>::type>
auto
pow(const Tensor<0, spacedim, ScalarType> &value, const SymbolicOp &op)
{
  static_assert(spacedim == SymbolicOp::space_dimension,
                "Incompatible spatial dimensions.");
  return pow(WeakForms::constant_tensor<0, spacedim>(value), op);
}


WEAK_FORMS_NAMESPACE_CLOSE


#endif // dealii_weakforms_mixed_operators_h