#ifndef _weak_forms_config_h
#define _weak_forms_config_h

#include <deal.II/base/config.h>

#define WEAK_FORMS_NAMESPACE_NAME dealiiWeakForms

#define WEAK_FORMS_NAMESPACE_OPEN     \
  namespace WEAK_FORMS_NAMESPACE_NAME \
  {
#define WEAK_FORMS_NAMESPACE_CLOSE } // namespace dealiiWeakForms


// Unconditionally import the deal.II namespace
WEAK_FORMS_NAMESPACE_OPEN
using namespace dealii;
WEAK_FORMS_NAMESPACE_CLOSE

#endif // _weak_forms_config_h