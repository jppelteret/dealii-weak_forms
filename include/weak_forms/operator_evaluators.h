#ifndef dealii_weakforms_operator_evaluators_h
#define dealii_weakforms_operator_evaluators_h

#include <deal.II/base/config.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/meshworker/scratch_data.h>

#include <weak_forms/config.h>
#include <weak_forms/type_traits.h>
#include <weak_forms/types.h>

#include <string>
#include <type_traits>
#include <vector>



WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  namespace Operators
  {
    namespace internal
    {
      // These operator evaluators are implemented to help evaluate complex
      // expressions that comprise an expression tree.
      // They are decomposes into units that know how to evaluate specific
      // components of the tree.
      // This helps prevent logic explosion as the number of permutations of
      // valid input types increases.
      // We do our best to guard against possible misuse by limiting the defined
      // operators to only valid ones, and guarding the various evaluators
      // using type trait checks.


      // Forward declarations

      /**
       * A struct that evaluates symbolic operations.
       *
       * These represent the terminal points on the expression tree.
       */
      template <typename OpType, typename T = void>
      struct SymbolicOpEvaluator;

      /**
       * A struct that evaluates unary operations.
       *
       * These represent the non-branching nodes on the expression tree.
       */
      template <typename OpType, typename T = void>
      struct UnaryOpEvaluator;

      /**
       * A struct that evaluates binary operations.
       *
       * These represent the branch points on the expression tree.
       */
      template <typename LhsOpType, typename RhsOpType, typename T = void>
      struct BinaryOpEvaluator;

      /**
       * A struct to help us which evaluator to use to evaluate a branch or
       * leaf operation, the result of which is then to be used to evaluate
       * the current operator.
       */
      template <typename OpType, typename U = void>
      struct BranchEvaluator;


      // ---- SYMBOLIC OPERATORS -----
      // These represent the terminal points on the expression tree.

      template <typename OpType>
      struct SymbolicOpEvaluator<
        OpType,
        typename std::enable_if<
          !is_test_function_or_trial_solution_op<OpType>::value &&
          !is_evaluated_with_scratch_data<OpType>::value>::type>
      {
        template <typename ScalarType, typename FEValuesType>
        static typename OpType::template return_type<ScalarType>
        apply(const OpType &operand, const FEValuesType &fe_values)
        {
          return operand.template operator()<ScalarType>(fe_values);
        }

        template <typename ScalarType,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp>
        static typename OpType::template return_type<ScalarType>
        apply(const OpType &          operand,
              const FEValuesTypeDoFs &fe_values_dofs,
              const FEValuesTypeOp &  fe_values_op)
        {
          (void)fe_values_dofs;

          // Delegate to the other function.
          return apply<ScalarType>(operand, fe_values_op);
        }

        template <typename ScalarType,
                  typename FEValuesType,
                  int dim,
                  int spacedim>
        static typename OpType::template return_type<ScalarType>
        apply(const OpType &                          operand,
              const FEValuesType &                    fe_values,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names)
        {
          (void)scratch_data;
          (void)solution_names;

          // Delegate to the other function.
          return apply<ScalarType>(operand, fe_values);
        }

        template <typename ScalarType,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp,
                  int dim,
                  int spacedim>
        static typename OpType::template return_type<ScalarType>
        apply(const OpType &                          operand,
              const FEValuesTypeDoFs &                fe_values_dofs,
              const FEValuesTypeOp &                  fe_values_op,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names)
        {
          (void)fe_values_dofs;
          (void)scratch_data;
          (void)solution_names;

          // Delegate to the other function.
          return apply<ScalarType>(operand, fe_values_op);
        }

        // ----- VECTORIZATION -----

        template <typename ScalarType, std::size_t width, typename FEValuesType>
        static
          typename OpType::template vectorized_return_type<ScalarType, width>
          apply(const OpType &                      operand,
                const FEValuesType &                fe_values,
                const types::vectorized_qp_range_t &q_point_range)
        {
          return operand.template operator()<ScalarType, width>(fe_values,
                                                                q_point_range);
        }

        template <typename ScalarType,
                  std::size_t width,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp>
        static
          typename OpType::template vectorized_return_type<ScalarType, width>
          apply(const OpType &                      operand,
                const FEValuesTypeDoFs &            fe_values_dofs,
                const FEValuesTypeOp &              fe_values_op,
                const types::vectorized_qp_range_t &q_point_range)
        {
          (void)fe_values_dofs;

          // Delegate to the other function.
          return apply<ScalarType, width>(operand, fe_values_op, q_point_range);
        }

        template <typename ScalarType,
                  std::size_t width,
                  typename FEValuesType,
                  int dim,
                  int spacedim>
        static
          typename OpType::template vectorized_return_type<ScalarType, width>
          apply(const OpType &                          operand,
                const FEValuesType &                    fe_values,
                MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                const std::vector<std::string> &        solution_names,
                const types::vectorized_qp_range_t &    q_point_range)
        {
          (void)scratch_data;
          (void)solution_names;

          // Delegate to the other function.
          return apply<ScalarType, width>(operand, fe_values, q_point_range);
        }

        template <typename ScalarType,
                  std::size_t width,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp,
                  int dim,
                  int spacedim>
        static
          typename OpType::template vectorized_return_type<ScalarType, width>
          apply(const OpType &                          operand,
                const FEValuesTypeDoFs &                fe_values_dofs,
                const FEValuesTypeOp &                  fe_values_op,
                MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                const std::vector<std::string> &        solution_names,
                const types::vectorized_qp_range_t &    q_point_range)
        {
          (void)fe_values_dofs;
          (void)scratch_data;
          (void)solution_names;

          // Delegate to the other function.
          return apply<ScalarType, width>(operand, fe_values_op, q_point_range);
        }
      };

      template <typename OpType>
      struct SymbolicOpEvaluator<
        OpType,
        typename std::enable_if<
          !is_test_function_or_trial_solution_op<OpType>::value &&
          is_evaluated_with_scratch_data<OpType>::value>::type>
      {
        template <typename ScalarType, int dim, int spacedim>
        static typename OpType::template return_type<ScalarType>
        apply(const OpType &                          operand,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names)
        {
          return operand.template operator()<ScalarType>(scratch_data,
                                                         solution_names);
        }

        template <typename ScalarType,
                  typename FEValuesType,
                  int dim,
                  int spacedim>
        static typename OpType::template return_type<ScalarType>
        apply(const OpType &                          operand,
              const FEValuesType &                    fe_values,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names)
        {
          (void)fe_values;

          // Delegate to the other function.
          return apply<ScalarType>(operand, scratch_data, solution_names);
        }

        template <typename ScalarType,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp,
                  int dim,
                  int spacedim>
        static typename OpType::template return_type<ScalarType>
        apply(const OpType &                          operand,
              const FEValuesTypeDoFs &                fe_values_dofs,
              const FEValuesTypeOp &                  fe_values_op,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names)
        {
          (void)fe_values_dofs;
          (void)fe_values_op;

          // Delegate to the other function.
          return apply<ScalarType>(operand, scratch_data, solution_names);
        }

        // ----- VECTORIZATION -----

        template <typename ScalarType, std::size_t width, int dim, int spacedim>
        static
          typename OpType::template vectorized_return_type<ScalarType, width>
          apply(const OpType &                          operand,
                MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                const std::vector<std::string> &        solution_names,
                const types::vectorized_qp_range_t &    q_point_range)
        {
          return operand.template operator()<ScalarType, width>(scratch_data,
                                                                solution_names,
                                                                q_point_range);
        }

        template <typename ScalarType,
                  std::size_t width,
                  typename FEValuesType,
                  int dim,
                  int spacedim>
        static
          typename OpType::template vectorized_return_type<ScalarType, width>
          apply(const OpType &                          operand,
                const FEValuesType &                    fe_values,
                MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                const std::vector<std::string> &        solution_names,
                const types::vectorized_qp_range_t &    q_point_range)
        {
          (void)fe_values;

          // Delegate to the other function.
          return apply<ScalarType, width>(operand,
                                          scratch_data,
                                          solution_names,
                                          q_point_range);
        }

        template <typename ScalarType,
                  std::size_t width,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp,
                  int dim,
                  int spacedim>
        static
          typename OpType::template vectorized_return_type<ScalarType, width>
          apply(const OpType &                          operand,
                const FEValuesTypeDoFs &                fe_values_dofs,
                const FEValuesTypeOp &                  fe_values_op,
                MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                const std::vector<std::string> &        solution_names,
                const types::vectorized_qp_range_t &    q_point_range)
        {
          (void)fe_values_dofs;
          (void)fe_values_op;

          // Delegate to the other function.
          return apply<ScalarType, width>(operand,
                                          scratch_data,
                                          solution_names,
                                          q_point_range);
        }
      };

      template <typename OpType>
      struct SymbolicOpEvaluator<
        OpType,
        typename std::enable_if<
          is_test_function_or_trial_solution_op<OpType>::value>::type>
      {
        static_assert(
          !is_evaluated_with_scratch_data<OpType>::value,
          "Expect the test function or trial solution operator not to be evaluated using scratch data.");

        template <typename ScalarType, typename FEValuesType>
        static typename OpType::template value_type<ScalarType>
        apply(const OpType &      operand,
              const FEValuesType &fe_values,
              const unsigned int  dof_index)
        {
          return operand.template operator()<ScalarType>(fe_values, dof_index);
        }

        template <typename ScalarType,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp>
        static typename OpType::template return_type<ScalarType>
        apply(const OpType &          operand,
              const FEValuesTypeDoFs &fe_values_dofs,
              const FEValuesTypeOp &  fe_values_op)
        {
          return operand.template operator()<ScalarType>(fe_values_dofs,
                                                         fe_values_op);
        }

        template <typename ScalarType,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp,
                  int dim,
                  int spacedim>
        static typename OpType::template return_type<ScalarType>
        apply(const OpType &                          operand,
              const FEValuesTypeDoFs &                fe_values_dofs,
              const FEValuesTypeOp &                  fe_values_op,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names)
        {
          (void)scratch_data;
          (void)solution_names;

          // Delegate to the other function.
          return apply<ScalarType>(operand, fe_values_dofs, fe_values_op);
        }

        // ----- VECTORIZATION -----

        template <typename ScalarType, std::size_t width, typename FEValuesType>
        static
          typename OpType::template vectorized_return_type<ScalarType, width>
          apply(const OpType &                      operand,
                const FEValuesType &                fe_values,
                const unsigned int                  dof_index,
                const types::vectorized_qp_range_t &q_point_range)
        {
          return operand.template operator()<ScalarType, width>(fe_values,
                                                                dof_index,
                                                                q_point_range);
        }

        template <typename ScalarType,
                  std::size_t width,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp>
        static
          typename OpType::template vectorized_return_type<ScalarType, width>
          apply(const OpType &                      operand,
                const FEValuesTypeDoFs &            fe_values_dofs,
                const FEValuesTypeOp &              fe_values_op,
                const types::vectorized_qp_range_t &q_point_range)
        {
          return operand.template operator()<ScalarType, width>(fe_values_dofs,
                                                                fe_values_op,
                                                                q_point_range);
        }

        template <typename ScalarType,
                  std::size_t width,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp,
                  int dim,
                  int spacedim>
        static
          typename OpType::template vectorized_return_type<ScalarType, width>
          apply(const OpType &                          operand,
                const FEValuesTypeDoFs &                fe_values_dofs,
                const FEValuesTypeOp &                  fe_values_op,
                MeshWorker::ScratchData<dim, spacedim> &scratch_data,
                const std::vector<std::string> &        solution_names,
                const types::vectorized_qp_range_t &    q_point_range)
        {
          (void)scratch_data;
          (void)solution_names;

          // Delegate to the other function.
          return apply<ScalarType, width>(operand,
                                          fe_values_dofs,
                                          fe_values_op,
                                          q_point_range);
        }
      };


      // ---- UNARY OPERATORS -----
      // These represent the non-branching nodes on the expression tree.



      // ---- Operators NOT for test functions / trial solutions ---
      // So these are restricted to symbolic ops, functors (standard and
      // cache)  and field solutions as leaf operations.


      /**
       * Helper to return values at all quadrature points
       *
       * Specialization: The operand is neither a test function nor trial
       * solution.
       *
       * Specialization: The operand is a neither field solution nor a
       * cache functor.
       */
      template <typename OpType>
      struct UnaryOpEvaluator<
        OpType,
        typename std::enable_if<
          !is_or_has_test_function_or_trial_solution_op<OpType>::value &&
          !is_or_has_evaluated_with_scratch_data<OpType>::value>::type>
      {
        template <typename ScalarType,
                  typename UnaryOpType,
                  typename FEValuesType>
        static typename UnaryOpType::template return_type<ScalarType>
        apply(const UnaryOpType & op,
              const OpType &      operand,
              const FEValuesType &fe_values)
        {
          // return op.template operator()<ScalarType>(
          //   operand.template operator()<ScalarType>(fe_values));

          return op.template operator()<ScalarType>(
            BranchEvaluator<OpType>::template evaluate<ScalarType>(operand,
                                                                   fe_values));
        }

        template <typename ScalarType,
                  typename UnaryOpType,
                  typename FEValuesType,
                  int dim,
                  int spacedim>
        static typename UnaryOpType::template return_type<ScalarType>
        apply(const UnaryOpType &                     op,
              const OpType &                          operand,
              const FEValuesType &                    fe_values,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names)
        {
          (void)scratch_data;
          (void)solution_names;

          // Delegate to the other function.
          return apply<ScalarType>(op, operand, fe_values);
        }

        // template <typename ScalarType,
        //           typename UnaryOpType,
        //           int dim,
        //           int spacedim>
        // static typename UnaryOpType::template return_type<ScalarType>
        // apply(const UnaryOpType &                     op,
        //       const OpType &                          operand,
        //       const FEValuesBase<dim, spacedim> &     fe_values_dofs,
        //       const FEValuesBase<dim, spacedim> &     fe_values_op,
        //       MeshWorker::ScratchData<dim, spacedim> &scratch_data,
        //       const std::vector<std::string> &        solution_names)
        // {
        //   (void)fe_values_dofs;
        //   (void)scratch_data;
        //   (void)solution_names;

        //   // Delegate to the other function.
        //   return apply<ScalarType>(op, operand, fe_values_op);
        // }

        // ----- VECTORIZATION -----

        template <typename ScalarType,
                  std::size_t width,
                  typename UnaryOpType,
                  typename FEValuesType>
        static typename UnaryOpType::template vectorized_return_type<ScalarType,
                                                                     width>
        apply(const UnaryOpType &                 op,
              const OpType &                      operand,
              const FEValuesType &                fe_values,
              const types::vectorized_qp_range_t &q_point_range)
        {
          return op.template operator()<ScalarType, width>(
            BranchEvaluator<OpType>::template evaluate<ScalarType, width>(
              operand, fe_values, q_point_range));
        }

        template <typename ScalarType,
                  std::size_t width,
                  typename UnaryOpType,
                  typename FEValuesType,
                  int dim,
                  int spacedim>
        static typename UnaryOpType::template vectorized_return_type<ScalarType,
                                                                     width>
        apply(const UnaryOpType &                     op,
              const OpType &                          operand,
              const FEValuesType &                    fe_values,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names,
              const types::vectorized_qp_range_t &    q_point_range)
        {
          (void)scratch_data;
          (void)solution_names;

          // Delegate to the other function.
          return apply<ScalarType, width>(op,
                                          operand,
                                          fe_values,
                                          q_point_range);
        }
      };


      /**
       * Helper to return values at all quadrature points
       *
       * Specialization: The operand is neither a test function nor trial
       * solution.
       *
       * Specialization: The operand is a either field solution or a
       * cache functor.
       */
      template <typename OpType>
      struct UnaryOpEvaluator<
        OpType,
        typename std::enable_if<
          !is_or_has_test_function_or_trial_solution_op<OpType>::value &&
          is_or_has_evaluated_with_scratch_data<OpType>::value>::type>
      {
        template <typename ScalarType,
                  typename UnaryOpType,
                  int dim,
                  int spacedim>
        static typename UnaryOpType::template return_type<ScalarType>
        apply(const UnaryOpType &                     op,
              const OpType &                          operand,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names)
        {
          // return op.template operator()<ScalarType>(
          //   operand.template operator()<ScalarType>(scratch_data,
          //                                           solution_names));

          return op.template operator()<ScalarType>(
            BranchEvaluator<OpType>::template evaluate<ScalarType>(
              operand, scratch_data, solution_names));
        }

        template <typename ScalarType,
                  typename UnaryOpType,
                  typename FEValuesType,
                  int dim,
                  int spacedim>
        static typename UnaryOpType::template return_type<ScalarType>
        apply(const UnaryOpType &                     op,
              const OpType &                          operand,
              const FEValuesType &                    fe_values,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names)
        {
          // Might be a compound op, so cannot split off the fe_values
          // object as it might be needed to evaluate another branch of
          // the tree.
          // return op.template operator()<ScalarType>(
          //   operand.template operator()<ScalarType>(fe_values,
          //                                           scratch_data,
          //                                           solution_names));

          return op.template operator()<ScalarType>(
            BranchEvaluator<OpType>::template evaluate<ScalarType>(
              operand, fe_values, scratch_data, solution_names));
        }

        template <typename ScalarType,
                  typename UnaryOpType,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp,
                  int dim,
                  int spacedim>
        static typename UnaryOpType::template return_type<ScalarType>
        apply(const UnaryOpType &                     op,
              const OpType &                          operand,
              const FEValuesTypeDoFs &                fe_values_dofs,
              const FEValuesTypeOp &                  fe_values_op,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names)
        {
          (void)fe_values_dofs;

          return op.template operator()<ScalarType>(
            BranchEvaluator<OpType>::template evaluate<ScalarType>(
              operand, fe_values_op, scratch_data, solution_names));
        }

        // ----- VECTORIZATION -----

        template <typename ScalarType,
                  std::size_t width,
                  typename UnaryOpType,
                  int dim,
                  int spacedim>
        static typename UnaryOpType::template vectorized_return_type<ScalarType,
                                                                     width>
        apply(const UnaryOpType &                     op,
              const OpType &                          operand,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names,
              const types::vectorized_qp_range_t &    q_point_range)
        {
          return op.template operator()<ScalarType, width>(
            BranchEvaluator<OpType>::template evaluate<ScalarType, width>(
              operand, scratch_data, solution_names, q_point_range));
        }

        template <typename ScalarType,
                  std::size_t width,
                  typename UnaryOpType,
                  typename FEValuesType,
                  int dim,
                  int spacedim>
        static typename UnaryOpType::template vectorized_return_type<ScalarType,
                                                                     width>
        apply(const UnaryOpType &                     op,
              const OpType &                          operand,
              const FEValuesType &                    fe_values,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names,
              const types::vectorized_qp_range_t &    q_point_range)
        {
          return op.template operator()<ScalarType, width>(
            BranchEvaluator<OpType>::template evaluate<ScalarType, width>(
              operand, fe_values, scratch_data, solution_names, q_point_range));
        }

        template <typename ScalarType,
                  std::size_t width,
                  typename UnaryOpType,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp,
                  int dim,
                  int spacedim>
        static typename UnaryOpType::template vectorized_return_type<ScalarType,
                                                                     width>
        apply(const UnaryOpType &                     op,
              const OpType &                          operand,
              const FEValuesTypeDoFs &                fe_values_dofs,
              const FEValuesTypeOp &                  fe_values_op,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names,
              const types::vectorized_qp_range_t &    q_point_range)
        {
          (void)fe_values_dofs;

          return op.template operator()<ScalarType, width>(
            BranchEvaluator<OpType>::template evaluate<ScalarType, width>(
              operand,
              fe_values_op,
              scratch_data,
              solution_names,
              q_point_range));
        }
      };


      // ---- Operators for test functions / trial solutions ---
      // So these are for when a test function or trial solution is one or more
      // of the leaf operations. The other leaves may or may not be
      // symbolic ops, functors (standard and cache) and field solutions.


      /**
       * Helper to return values for all DoFs, evaluated at one quadrature
       * point.
       *
       * Specialization: The operand is either a test function or trial
       * solution.
       *
       * Specialization: The operand is a neither field solution nor a
       * cache functor.
       */
      template <typename OpType>
      struct UnaryOpEvaluator<
        OpType,
        typename std::enable_if<
          is_or_has_test_function_or_trial_solution_op<OpType>::value &&
          !is_or_has_evaluated_with_scratch_data<OpType>::value>::type>
      {
        template <typename ScalarType,
                  typename UnaryOpType,
                  typename FEValuesType>
        static typename UnaryOpType::template value_type<ScalarType>
        apply(const UnaryOpType & op,
              const OpType &      operand,
              const FEValuesType &fe_values,
              const unsigned int  dof_index)
        {
          return op.template operator()<ScalarType>(
            BranchEvaluator<OpType>::template evaluate<ScalarType>(operand,
                                                                   fe_values,
                                                                   dof_index));
        }

        template <typename ScalarType,
                  typename UnaryOpType,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp>
        static typename UnaryOpType::template return_type<ScalarType>
        apply(const UnaryOpType &     op,
              const OpType &          operand,
              const FEValuesTypeDoFs &fe_values_dofs,
              const FEValuesTypeOp &  fe_values_op)
        {
          // return op.template operator()<ScalarType>(
          //   operand.template operator()<ScalarType>(fe_values_dofs,
          //                                           fe_values_op));

          return op.template operator()<ScalarType>(
            BranchEvaluator<OpType>::template evaluate<ScalarType>(
              operand, fe_values_dofs, fe_values_op));
        }

        template <typename ScalarType,
                  typename UnaryOpType,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp,
                  int dim,
                  int spacedim>
        static typename UnaryOpType::template return_type<ScalarType>
        apply(const UnaryOpType &                     op,
              const OpType &                          operand,
              const FEValuesTypeDoFs &                fe_values_dofs,
              const FEValuesTypeOp &                  fe_values_op,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names)
        {
          (void)scratch_data;
          (void)solution_names;

          return op.template operator()<ScalarType>(
            BranchEvaluator<OpType>::template evaluate<ScalarType>(
              operand, fe_values_dofs, fe_values_op));
        }

        // template <typename ScalarType,
        //           typename UnaryOpType,
        //           int dim,
        //           int spacedim>
        // static typename UnaryOpType::template return_type<ScalarType>
        // apply(const UnaryOpType &                     op,
        //       const OpType &                          operand,
        //       const FEValuesBase<dim, spacedim> &     fe_values,
        //       MeshWorker::ScratchData<dim, spacedim> &scratch_data,
        //       const std::vector<std::string> &        solution_names,
        //          const unsigned int                              dof_index)
        // {
        //   (void)scratch_data;
        //   (void)solution_names;

        //   // Delegate to the other function.
        //   return apply<ScalarType>(op, operand, fe_values, dof_index);
        // }

        // ----- VECTORIZATION -----

        template <typename ScalarType,
                  std::size_t width,
                  typename UnaryOpType,
                  typename FEValuesType>
        static typename UnaryOpType::template vectorized_return_type<ScalarType,
                                                                     width>
        apply(const UnaryOpType &                 op,
              const OpType &                      operand,
              const FEValuesType &                fe_values,
              const unsigned int                  dof_index,
              const types::vectorized_qp_range_t &q_point_range)
        {
          return op.template operator()<ScalarType, width>(
            BranchEvaluator<OpType>::template evaluate<ScalarType, width>(
              operand, fe_values, dof_index, q_point_range));
        }

        template <typename ScalarType,
                  std::size_t width,
                  typename UnaryOpType,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp>
        static typename UnaryOpType::template vectorized_return_type<ScalarType,
                                                                     width>
        apply(const UnaryOpType &                 op,
              const OpType &                      operand,
              const FEValuesTypeDoFs &            fe_values_dofs,
              const FEValuesTypeOp &              fe_values_op,
              const types::vectorized_qp_range_t &q_point_range)
        {
          return op.template operator()<ScalarType, width>(
            BranchEvaluator<OpType>::template evaluate<ScalarType, width>(
              operand, fe_values_dofs, fe_values_op, q_point_range));
        }

        template <typename ScalarType,
                  std::size_t width,
                  typename UnaryOpType,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp,
                  int dim,
                  int spacedim>
        static typename UnaryOpType::template vectorized_return_type<ScalarType,
                                                                     width>
        apply(const UnaryOpType &                     op,
              const OpType &                          operand,
              const FEValuesTypeDoFs &                fe_values_dofs,
              const FEValuesTypeOp &                  fe_values_op,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names,
              const types::vectorized_qp_range_t &    q_point_range)
        {
          (void)scratch_data;
          (void)solution_names;

          return op.template operator()<ScalarType, width>(
            BranchEvaluator<OpType>::template evaluate<ScalarType, width>(
              operand, fe_values_dofs, fe_values_op, q_point_range));
        }
      };


      /**
       * Helper to return values at all quadrature points
       *
       * Specialization: The operand is either a test function or trial
       solution.
       * Specialization: The operand is a either field solution or a cache
       * functor.
       */
      template <typename OpType>
      struct UnaryOpEvaluator<
        OpType,
        typename std::enable_if<
          is_or_has_test_function_or_trial_solution_op<OpType>::value &&
          is_or_has_evaluated_with_scratch_data<OpType>::value>::type>
      {
        template <typename ScalarType,
                  typename UnaryOpType,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp,
                  int dim,
                  int spacedim>
        static typename UnaryOpType::template return_type<ScalarType>
        apply(const UnaryOpType &                     op,
              const OpType &                          operand,
              const FEValuesTypeDoFs &                fe_values_dofs,
              const FEValuesTypeOp &                  fe_values_op,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names)
        {
          return op.template operator()<ScalarType>(
            BranchEvaluator<OpType>::template evaluate<ScalarType>(
              operand,
              fe_values_dofs,
              fe_values_op,
              scratch_data,
              solution_names));
        }

        // ----- VECTORIZATION -----

        template <typename ScalarType,
                  std::size_t width,
                  typename UnaryOpType,
                  typename FEValuesTypeDoFs,
                  typename FEValuesTypeOp,
                  int dim,
                  int spacedim>
        static typename UnaryOpType::template vectorized_return_type<ScalarType,
                                                                     width>
        apply(const UnaryOpType &                     op,
              const OpType &                          operand,
              const FEValuesTypeDoFs &                fe_values_dofs,
              const FEValuesTypeOp &                  fe_values_op,
              MeshWorker::ScratchData<dim, spacedim> &scratch_data,
              const std::vector<std::string> &        solution_names,
              const types::vectorized_qp_range_t &    q_point_range)
        {
          return op.template operator()<ScalarType, width>(
            BranchEvaluator<OpType>::template evaluate<ScalarType, width>(
              operand,
              fe_values_dofs,
              fe_values_op,
              scratch_data,
              solution_names,
              q_point_range));
        }
      };


      // ---- BINARY OPERATORS -----
      // These represent the branch points on the expression tree.


      /**
       * Helper to return values at all quadrature points
       *
       * A one-size-fits-all solution to the combinatorical problem:
       * is_or_has_test_function_or_trial_solution_op<LhsOpType>::value ==
       * [true/false] &&
       * is_or_has_test_function_or_trial_solution_op<RhsOpType>::value ==
       * [true/false] && is_or_has_evaluated_with_scratch_data<LhsOpType>::value
       * == [true/false] &&
       * is_or_has_evaluated_with_scratch_data<RhsOpType>::value == [true/false]
       */
      template <typename LhsOpType, typename RhsOpType>
      struct BinaryOpEvaluator<LhsOpType, RhsOpType>
      {
        template <typename ScalarType,
                  typename BinaryOpType,
                  typename... Arguments>
        static typename BinaryOpType::template return_type<ScalarType>
        apply(const BinaryOpType &op,
              const LhsOpType &   lhs_operand,
              const RhsOpType &   rhs_operand,
              Arguments &...args)
        {
          return op.template operator()<ScalarType>(
            BranchEvaluator<LhsOpType>::template evaluate<ScalarType>(
              lhs_operand, args...),
            BranchEvaluator<RhsOpType>::template evaluate<ScalarType>(
              rhs_operand, args...));
          // lhs_operand.template operator()<ScalarType>(args...),
          // rhs_operand.template operator()<ScalarType>(args...));
        }

        // ----- VECTORIZATION -----

        template <typename ScalarType,
                  std::size_t width,
                  typename BinaryOpType,
                  typename... Arguments>
        static
          typename BinaryOpType::template vectorized_return_type<ScalarType,
                                                                 width>
          apply(const BinaryOpType &op,
                const LhsOpType &   lhs_operand,
                const RhsOpType &   rhs_operand,
                Arguments &...args)
        {
          return op.template operator()<ScalarType, width>(
            BranchEvaluator<LhsOpType>::template evaluate<ScalarType, width>(
              lhs_operand, args...),
            BranchEvaluator<RhsOpType>::template evaluate<ScalarType, width>(
              rhs_operand, args...));
        }
      };


      // // ---- Operators NOT for test functions / trial solutions ---
      // // So these are restricted to symbolic ops, functors (standard and
      // // cache)  and field solutions as leaf operations.


      // /**
      //  * Helper to return values at all quadrature points
      //  *
      //  * Specialization: Neither operand is a field solution or a cache
      //  functor.
      //  */
      // template <typename LhsOpType, typename RhsOpType>
      // struct BinaryOpEvaluator<
      //   LhsOpType,
      //   RhsOpType,
      //   typename std::enable_if<
      //     !is_or_has_test_function_or_trial_solution_op<LhsOpType>::value &&
      //     !is_or_has_test_function_or_trial_solution_op<RhsOpType>::value &&
      //     !is_or_has_evaluated_with_scratch_data<LhsOpType>::value &&
      //     !is_or_has_evaluated_with_scratch_data<RhsOpType>::value>::type>
      // {
      //   template <typename ScalarType,
      //             typename BinaryOpType,
      //             int dim,
      //             int spacedim>
      //   static typename BinaryOpType::template return_type<ScalarType>
      //   apply(const BinaryOpType &               op,
      //         const LhsOpType &                  lhs_operand,
      //         const RhsOpType &                  rhs_operand,
      //         const FEValuesType &fe_values)
      //   {
      //     return op.template operator()<ScalarType>(
      //       lhs_operand.template operator()<ScalarType>(fe_values),
      //       rhs_operand.template operator()<ScalarType>(fe_values));
      //   }

      //   template <typename ScalarType,
      //             typename BinaryOpType,
      //             int dim,
      //             int spacedim>
      //   static typename BinaryOpType::template return_type<ScalarType>
      //   apply(const BinaryOpType &                    op,
      //         const LhsOpType &                       lhs_operand,
      //         const RhsOpType &                       rhs_operand,
      //         const FEValuesBase<dim, spacedim> &     fe_values,
      //         MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      //         const std::vector<std::string> &        solution_names)
      //   {
      //     (void)scratch_data;
      //     (void)solution_names;

      //     // Delegate to the other function.
      //     return apply<ScalarType>(op, lhs_operand, rhs_operand, fe_values);
      //   }
      // };


      // /**
      //  * Helper to return values at all quadrature points
      //  *
      //  * Specialization: LHS operand is a field solution or a cache functor.
      //  */
      // template <typename LhsOpType, typename RhsOpType>
      // struct BinaryOpEvaluator<
      //   LhsOpType,
      //   RhsOpType,
      //   typename std::enable_if<
      //     !is_or_has_test_function_or_trial_solution_op<LhsOpType>::value &&
      //     !is_or_has_test_function_or_trial_solution_op<RhsOpType>::value &&
      //     is_or_has_evaluated_with_scratch_data<LhsOpType>::value &&
      //     !is_or_has_evaluated_with_scratch_data<RhsOpType>::value>::type>
      // {
      //   template <typename ScalarType,
      //             typename BinaryOpType,
      //             int dim,
      //             int spacedim>
      //   static typename BinaryOpType::template return_type<ScalarType>
      //   apply(const BinaryOpType &                    op,
      //         const LhsOpType &                       lhs_operand,
      //         const RhsOpType &                       rhs_operand,
      //         const FEValuesBase<dim, spacedim> &     fe_values,
      //         MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      //         const std::vector<std::string> &        solution_names)
      //   {
      //     return op.template operator()<ScalarType>(
      //       lhs_operand.template operator()<ScalarType>(scratch_data,
      //                                                   solution_names),
      //       rhs_operand.template operator()<ScalarType>(fe_values));
      //   }
      // };


      // /**
      //  * Helper to return values at all quadrature points
      //  *
      //  * Specialization: RHS operand is a field solution or a cache functor.
      //  */
      // template <typename LhsOpType, typename RhsOpType>
      // struct BinaryOpEvaluator<
      //   LhsOpType,
      //   RhsOpType,
      //   typename std::enable_if<
      //     !is_or_has_test_function_or_trial_solution_op<LhsOpType>::value &&
      //     !is_or_has_test_function_or_trial_solution_op<RhsOpType>::value &&
      //     !is_or_has_evaluated_with_scratch_data<LhsOpType>::value &&
      //     is_or_has_evaluated_with_scratch_data<RhsOpType>::value>::type>
      // {
      //   template <typename ScalarType,
      //             typename BinaryOpType,
      //             int dim,
      //             int spacedim>
      //   static typename BinaryOpType::template return_type<ScalarType>
      //   apply(const BinaryOpType &                    op,
      //         const LhsOpType &                       lhs_operand,
      //         const RhsOpType &                       rhs_operand,
      //         const FEValuesBase<dim, spacedim> &     fe_values,
      //         MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      //         const std::vector<std::string> &        solution_names)
      //   {
      //     return op.template operator()<ScalarType>(
      //       lhs_operand.template operator()<ScalarType>(fe_values),
      //       rhs_operand.template operator()<ScalarType>(scratch_data,
      //                                                   solution_names));
      //   }
      // };


      // /**
      //  * Helper to return values at all quadrature points
      //  *
      //  * Specialization: Both operands are either field solutions or cache
      //  * functors.
      //  */
      // template <typename LhsOpType, typename RhsOpType>
      // struct BinaryOpEvaluator<
      //   LhsOpType,
      //   RhsOpType,
      //   typename std::enable_if<
      //     !is_or_has_test_function_or_trial_solution_op<LhsOpType>::value &&
      //     !is_or_has_test_function_or_trial_solution_op<RhsOpType>::value &&
      //     is_or_has_evaluated_with_scratch_data<LhsOpType>::value &&
      //     is_or_has_evaluated_with_scratch_data<RhsOpType>::value>::type>
      // {
      //   template <typename ScalarType,
      //             typename BinaryOpType,
      //             int dim,
      //             int spacedim>
      //   static typename BinaryOpType::template return_type<ScalarType>
      //   apply(const BinaryOpType &                    op,
      //         const LhsOpType &                       lhs_operand,
      //         const RhsOpType &                       rhs_operand,
      //         const FEValuesBase<dim, spacedim> &     fe_values,
      //         MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      //         const std::vector<std::string> &        solution_names)
      //   {
      //     (void)fe_values;

      //     return op.template operator()<ScalarType>(
      //       lhs_operand.template operator()<ScalarType>(scratch_data,
      //                                                   solution_names),
      //       rhs_operand.template operator()<ScalarType>(scratch_data,
      //                                                   solution_names));
      //   }
      // };


      // // ---- Operators for test functions / trial solutions ---
      // // So these are for when a test function or trial solution is one or
      // more
      // // of the leaf operations. The other leaves may or may not be
      // // symbolic ops, functors (standard and cache) and field solutions.


      // // --- LHS OP Only ---

      // template <typename LhsOpType, typename RhsOpType>
      // struct BinaryOpEvaluator<
      //   LhsOpType,
      //   RhsOpType,
      //   typename std::enable_if<
      //     is_or_has_test_function_or_trial_solution_op<LhsOpType>::value &&
      //     !is_or_has_test_function_or_trial_solution_op<RhsOpType>::value &&
      //     !is_or_has_evaluated_with_scratch_data<LhsOpType>::value &&
      //     !is_or_has_evaluated_with_scratch_data<RhsOpType>::value>::type>
      // {
      //   template <typename ScalarType,
      //             typename BinaryOpType,
      //             int dim,
      //             int spacedim>
      //   static typename BinaryOpType::template return_type<ScalarType>
      //   apply(const BinaryOpType &               op,
      //         const LhsOpType &                  lhs_operand,
      //         const RhsOpType &                  rhs_operand,
      //         const FEValuesTypeDoFs &fe_values_dofs,
      //         const FEValuesTypeOp &fe_values_op)
      //   {
      //     return op.template operator()<ScalarType>(
      //       lhs_operand.template operator()<ScalarType>(fe_values_dofs,
      //                                                   fe_values_op),
      //       rhs_operand.template operator()<ScalarType>(fe_values_op));
      //   }
      // };


      // // --- RHS OP Only ---

      // template <typename LhsOpType, typename RhsOpType>
      // struct BinaryOpEvaluator<
      //   LhsOpType,
      //   RhsOpType,
      //   typename std::enable_if<
      //     !is_or_has_test_function_or_trial_solution_op<LhsOpType>::value &&
      //     is_or_has_test_function_or_trial_solution_op<RhsOpType>::value &&
      //     !is_or_has_evaluated_with_scratch_data<LhsOpType>::value &&
      //     !is_or_has_evaluated_with_scratch_data<RhsOpType>::value>::type>
      // {
      //   template <typename ScalarType,
      //             typename BinaryOpType,
      //             int dim,
      //             int spacedim>
      //   static typename BinaryOpType::template return_type<ScalarType>
      //   apply(const BinaryOpType &               op,
      //         const LhsOpType &                  lhs_operand,
      //         const RhsOpType &                  rhs_operand,
      //         const FEValuesTypeDoFs &fe_values_dofs,
      //         const FEValuesTypeOp &fe_values_op)
      //   {
      //     return op.template operator()<ScalarType>(
      //       lhs_operand.template operator()<ScalarType>(fe_values_op),
      //       rhs_operand.template operator()<ScalarType>(fe_values_dofs,
      //                                                   fe_values_op));
      //   }
      // };


      // // --- LHS OP and RHS OP ---


      // /**
      //  * Helper to return values at all quadrature points
      //  *
      //  * Specialization: Neither operand is a field solution or a cache
      //  functor.
      //  */
      // template <typename LhsOpType, typename RhsOpType>
      // struct BinaryOpEvaluator<
      //   LhsOpType,
      //   RhsOpType,
      //   typename std::enable_if<
      //     is_or_has_test_function_or_trial_solution_op<LhsOpType>::value &&
      //     is_or_has_test_function_or_trial_solution_op<RhsOpType>::value &&
      //     !is_or_has_evaluated_with_scratch_data<LhsOpType>::value &&
      //     !is_or_has_evaluated_with_scratch_data<RhsOpType>::value>::type>
      // {
      //   template <typename ScalarType,
      //             typename BinaryOpType,
      //             int dim,
      //             int spacedim>
      //   static typename BinaryOpType::template return_type<ScalarType>
      //   apply(const BinaryOpType &               op,
      //         const LhsOpType &                  lhs_operand,
      //         const RhsOpType &                  rhs_operand,
      //         const FEValuesTypeDoFs &fe_values_dofs,
      //         const FEValuesTypeOp &fe_values_op)
      //   {
      //     return op.template operator()<ScalarType>(
      //       lhs_operand.template operator()<ScalarType>(fe_values_dofs,
      //                                                   fe_values_op),
      //       rhs_operand.template operator()<ScalarType>(fe_values_dofs,
      //                                                   fe_values_op));
      //   }
      // };


      // /**
      //  * Helper to return values at all quadrature points
      //  *
      //  * Specialization: LHS operand is a field solution or a cache functor.
      //  */
      // template <typename LhsOpType, typename RhsOpType>
      // struct BinaryOpEvaluator<
      //   LhsOpType,
      //   RhsOpType,
      //   typename std::enable_if<
      //     is_or_has_test_function_or_trial_solution_op<LhsOpType>::value &&
      //     is_or_has_test_function_or_trial_solution_op<RhsOpType>::value &&
      //     is_or_has_evaluated_with_scratch_data<LhsOpType>::value &&
      //     !is_or_has_evaluated_with_scratch_data<RhsOpType>::value>::type>
      // {
      //   template <typename ScalarType,
      //             typename BinaryOpType,
      //             int dim,
      //             int spacedim>
      //   static typename BinaryOpType::template return_type<ScalarType>
      //   apply(const BinaryOpType &                    op,
      //         const LhsOpType &                       lhs_operand,
      //         const RhsOpType &                       rhs_operand,
      //         const FEValuesTypeDoFs &fe_values_dofs,
      //         const FEValuesTypeOp &fe_values_op,
      //         MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      //         const std::vector<std::string> &        solution_names)
      //   {
      //     return op.template operator()<ScalarType>(
      //       lhs_operand.template operator()<ScalarType>(fe_values_dofs,
      //                                                   fe_values_op,
      //                                                   scratch_data,
      //                                                   solution_names),
      //       rhs_operand.template operator()<ScalarType>(fe_values_dofs,
      //                                                   fe_values_op));
      //   }
      // };


      // /**
      //  * Helper to return values at all quadrature points
      //  *
      //  * Specialization: RHS operand is a field solution or a cache functor.
      //  */
      // template <typename LhsOpType, typename RhsOpType>
      // struct BinaryOpEvaluator<
      //   LhsOpType,
      //   RhsOpType,
      //   typename std::enable_if<
      //     is_or_has_test_function_or_trial_solution_op<LhsOpType>::value &&
      //     is_or_has_test_function_or_trial_solution_op<RhsOpType>::value &&
      //     !is_or_has_evaluated_with_scratch_data<LhsOpType>::value &&
      //     is_or_has_evaluated_with_scratch_data<RhsOpType>::value>::type>
      // {
      //   template <typename ScalarType,
      //             typename BinaryOpType,
      //             int dim,
      //             int spacedim>
      //   static typename BinaryOpType::template return_type<ScalarType>
      //   apply(const BinaryOpType &                    op,
      //         const LhsOpType &                       lhs_operand,
      //         const RhsOpType &                       rhs_operand,
      //         const FEValuesTypeDoFs &fe_values_dofs,
      //         const FEValuesTypeOp &fe_values_op,
      //         MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      //         const std::vector<std::string> &        solution_names)
      //   {
      //     return op.template operator()<ScalarType>(
      //       lhs_operand.template operator()<ScalarType>(fe_values_dofs,
      //                                                   fe_values_op),
      //       rhs_operand.template operator()<ScalarType>(fe_values_dofs,
      //                                                   fe_values_op,
      //                                                   scratch_data,
      //                                                   solution_names));
      //   }
      // };


      // /**
      //  * Helper to return values at all quadrature points
      //  *
      //  * Specialization: Both operands are either field solutions or cache
      //  * functors.
      //  */
      // template <typename LhsOpType, typename RhsOpType>
      // struct BinaryOpEvaluator<
      //   LhsOpType,
      //   RhsOpType,
      //   typename std::enable_if<
      //     is_or_has_test_function_or_trial_solution_op<LhsOpType>::value &&
      //     is_or_has_test_function_or_trial_solution_op<RhsOpType>::value &&
      //     is_or_has_evaluated_with_scratch_data<LhsOpType>::value &&
      //     is_or_has_evaluated_with_scratch_data<RhsOpType>::value>::type>
      // {
      //   template <typename ScalarType,
      //             typename BinaryOpType,
      //             int dim,
      //             int spacedim>
      //   static typename BinaryOpType::template return_type<ScalarType>
      //   apply(const BinaryOpType &                    op,
      //         const LhsOpType &                       lhs_operand,
      //         const RhsOpType &                       rhs_operand,
      //         const FEValuesTypeDoFs &fe_values_dofs,
      //         const FEValuesTypeOp &fe_values_op,
      //         MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      //         const std::vector<std::string> &        solution_names)
      //   {
      //     return op.template operator()<ScalarType>(
      //       lhs_operand.template operator()<ScalarType>(fe_values_dofs,
      //                                                   fe_values_op,
      //                                                   scratch_data,
      //                                                   solution_names),
      //       rhs_operand.template operator()<ScalarType>(fe_values_dofs,
      //                                                   fe_values_op,
      //                                                   scratch_data,
      //                                                   solution_names));
      //   }
      // };


      // ---- BRANCH EVALUATORS -----

      template <typename OpType>
      struct BranchEvaluator<
        OpType,
        typename std::enable_if<!is_unary_op<OpType>::value &&
                                !is_binary_op<OpType>::value>::type>
      {
        template <typename ScalarType, typename... Arguments>
        static auto
        evaluate(const OpType &op, Arguments &...args)
        {
          // Use a shim to make all symbolic operations compatible with the
          // input arguments.
          return SymbolicOpEvaluator<OpType>::template apply<ScalarType>(
            op, args...); // std::forward<Arguments>(args)...)
        }

        template <typename ScalarType, std::size_t width, typename... Arguments>
        static auto
        evaluate(const OpType &op, Arguments &...args)
        {
          // Use a shim to make all symbolic operations compatible with the
          // input arguments.
          return SymbolicOpEvaluator<OpType>::template apply<ScalarType, width>(
            op, args...); // std::forward<Arguments>(args)...)
        }
      };


      template <typename OpType>
      struct BranchEvaluator<
        OpType,
        typename std::enable_if<is_unary_op<OpType>::value>::type>
      {
        template <typename ScalarType, typename... Arguments>
        static auto
        evaluate(const OpType &op, Arguments &...args)
        {
          // These can evaluate themselves. They also provide some additional
          // checks that we'd bypass if we directly called
          //
          // return UnaryOpEvaluator<typename OpType::OpType>::template apply<
          //   ScalarType>(op, op.get_operand(),
          //   std::forward<Arguments>(args)...);
          //
          // So, propogate the same argument types down the chain.
          // return op.template operator()<ScalarType>(
          //    args...); // std::forward<Arguments>(args)...)

          return UnaryOpEvaluator<typename OpType::OpType>::template apply<
            ScalarType>(op, op.get_operand(), args...);
        }

        template <typename ScalarType, std::size_t width, typename... Arguments>
        static auto
        evaluate(const OpType &op, Arguments &...args)
        {
          return UnaryOpEvaluator<typename OpType::OpType>::
            template apply<ScalarType, width>(op, op.get_operand(), args...);
        }
      };


      template <typename OpType>
      struct BranchEvaluator<
        OpType,
        typename std::enable_if<is_binary_op<OpType>::value>::type>
      {
        template <typename ScalarType, typename... Arguments>
        static auto
        evaluate(const OpType &op, Arguments &...args)
        {
          // These can evaluate themselves. They also provide some additional
          // checks that we'd bypass if we directly called
          //
          // return BinaryOpEvaluator<typename OpType::LhsOpType,
          //                          typename OpType::RhsOpType>::
          //   template apply<ScalarType>(op,
          //                              op.get_lhs_operand(),
          //                              op.get_rhs_operand(),
          //                              std::forward<Arguments>(args)...);
          //
          // So propogate the same argument types down the chain.
          // return op.template operator()<ScalarType>(
          //    args...);

          return BinaryOpEvaluator<typename OpType::LhsOpType,
                                   typename OpType::RhsOpType>::
            template apply<ScalarType>(op,
                                       op.get_lhs_operand(),
                                       op.get_rhs_operand(),
                                       args...);
        }

        template <typename ScalarType, std::size_t width, typename... Arguments>
        static auto
        evaluate(const OpType &op, Arguments &...args)
        {
          return BinaryOpEvaluator<typename OpType::LhsOpType,
                                   typename OpType::RhsOpType>::
            template apply<ScalarType, width>(op,
                                              op.get_lhs_operand(),
                                              op.get_rhs_operand(),
                                              args...);
        }
      };

    } // namespace internal
  }   // namespace Operators
} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE


#endif // dealii_weakforms_operator_evaluators_h