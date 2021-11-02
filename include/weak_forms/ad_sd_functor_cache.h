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

#ifndef dealii_weakforms_ad_sd_functor_cache_h
#define dealii_weakforms_ad_sd_functor_cache_h

#include <deal.II/base/config.h>

#include <deal.II/algorithms/general_data_storage.h>

#include <deal.II/differentiation/ad.h>
#include <deal.II/differentiation/sd.h>

#include <deal.II/meshworker/scratch_data.h>

#include <weak_forms/ad_sd_functor_internal.h>


WEAK_FORMS_NAMESPACE_OPEN


namespace WeakForms
{
  /**
   * @brief Persistent data for AD and SD calculations. The idea is that this
   * cache is initialised and stored in a user class, so that is not created
   * and destroyed each time the assembly loop is performed. That way we can
   * perform expensive calculations only once per simulation, rather than
   * each time the assembly loop occurs.
   */
  class AD_SD_Functor_Cache
  {
  public:
    template <int dim, int spacedim>
    static void
    initialize(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
               const AD_SD_Functor_Cache *const        ad_sd_functor_cache)
    {
      GeneralDataStorage &scratch_cache =
        scratch_data.get_general_data_storage();

      // If the user has provided a persistent data then we can leverage that
      // for the cache. Otherwise, the general data storage provided by the
      // scratch data object can simply reference itself as the cache.
      if (ad_sd_functor_cache != nullptr)
        {
          GeneralDataStorage &cache =
            const_cast<GeneralDataStorage &>(ad_sd_functor_cache->cache);
          scratch_cache.add_unique_reference<GeneralDataStorage>(
            get_name_ad_sd_cache(), cache);
          scratch_cache.add_unique_copy<bool>(get_name_ad_sd_cache_flag(),
                                              true);
        }
      else
        {
          scratch_cache.add_unique_reference<GeneralDataStorage>(
            get_name_ad_sd_cache(), scratch_cache);
        }
    }

    template <int dim, int spacedim>
    static bool
    has_user_cache(MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      return scratch_data.stores_object_with_name(get_name_ad_sd_cache_flag());
    }

    template <int dim, int spacedim>
    static bool
    has_user_cache(const MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      return scratch_data.stores_object_with_name(get_name_ad_sd_cache_flag());
    }

    template <int dim, int spacedim>
    static GeneralDataStorage &
    get_cache(MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      GeneralDataStorage &scratch_cache =
        scratch_data.get_general_data_storage();

      return scratch_cache.get_object_with_name<GeneralDataStorage>(
        get_name_ad_sd_cache());
    }

    template <int dim, int spacedim>
    static const GeneralDataStorage &
    get_cache(const MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      const GeneralDataStorage &scratch_cache =
        scratch_data.get_general_data_storage();

      return scratch_cache.get_object_with_name<GeneralDataStorage>(
        get_name_ad_sd_cache());
    }

  private:
    GeneralDataStorage cache;

    static std::string
    get_name_ad_sd_cache()
    {
      return Operators::internal::get_deal_II_prefix() + "AD_SD_Functor_Cache";
    }

    static std::string
    get_name_ad_sd_cache_flag()
    {
      return get_name_ad_sd_cache() + "_Flag";
    }
  };
} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE


#endif // dealii_weakforms_ad_sd_functor_cache_h
