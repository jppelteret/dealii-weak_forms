
#include <deal.II/base/exceptions.h>

#include <regex>
#include <string>
#include <vector>


DeclException4(ExcMatrixEntriesNotEqual,
               int,
               int,
               double,
               double,
               << "Matrix entries are different (exemplar). "
               << "(R,C) = (" << arg1 << "," << arg2 << "). "
               << "Blessed value: " << arg3 << "; "
               << "Other value: " << arg4 << ".");

DeclException2(ExcIteratorRowIndexNotEqual,
               int,
               int,
               << "Iterator row index mismatch. "
               << "  Iterator 1: " << arg1 << "  Iterator 2: " << arg2);

DeclException2(ExcIteratorColumnIndexNotEqual,
               int,
               int,
               << "Iterator column index mismatch. "
               << "  Iterator 1: " << arg1 << "  Iterator 2: " << arg2);


DeclException3(ExcVectorEntriesNotEqual,
               int,
               double,
               double,
               << "Vector entries are different (exemplar). "
               << "(R) = (" << arg1 << "). "
               << "Blessed value: " << arg2 << "; "
               << "Other value: " << arg3 << ".");


std::string
strip_off_namespace(std::string demangled_type)
{
  const std::vector<std::string> names{
    "dealii::WeakForms::Operators::", "dealii::WeakForms::", "dealii::"};

  for (const auto &name : names)
    demangled_type = std::regex_replace(demangled_type, std::regex(name), "");

  return demangled_type;
}
