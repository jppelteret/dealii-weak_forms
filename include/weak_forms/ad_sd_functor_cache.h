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

#include <deal.II/base/multithread_info.h>
#include <deal.II/base/thread_management.h>

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
    // The queue_length matches that used by default for WorkStream::run(), and
    // hence mesh_loop().
    AD_SD_Functor_Cache(
      const unsigned int queue_length = 2 * MultithreadInfo::n_threads())
      : source_lock_and_cache(queue_length,
                              std::make_pair(false, GeneralDataStorage()))
    {}

    template <int dim, int spacedim>
    static void
    initialize(MeshWorker::ScratchData<dim, spacedim> &scratch_data,
               const AD_SD_Functor_Cache *const        ad_sd_functor_cache)
    {
      // If the user has provided a persistent data then we can leverage that
      // for the cache. Otherwise, the general data storage provided by the
      // scratch data object can simply reference itself as the cache.
      if (ad_sd_functor_cache == nullptr)
        return;

      GeneralDataStorage &scratch_cache =
        scratch_data.get_general_data_storage();

      // Register the user cache in a central and always accessible place
      AD_SD_Functor_Cache &source_cache =
        const_cast<AD_SD_Functor_Cache &>(*ad_sd_functor_cache);
      scratch_cache.add_unique_reference<AD_SD_Functor_Cache>(
        get_name_ad_sd_cache(), source_cache);

      // Check that all cache entries are unlocked
      Assert(source_cache.all_entries_unlocked(),
             ExcMessage("Cache entry is already locked upon initialisation."));
    }

    template <int dim, int spacedim>
    static GeneralDataStorage &
    check_out_source_cache_from_pool(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      if (has_user_cache(scratch_data))
        {
          // We must ensure that we do not try to evaluate from multiple threads
          // at once.
          AD_SD_Functor_Cache &source_cache = get_source_cache(scratch_data);
          const std::lock_guard<Threads::Mutex> lock(source_cache.mutex);

          // We use an infinite loop because we cannot be sure when the next
          // entry will become free for use.
          while (true)
            {
              for (auto &lock_and_cache : source_cache.source_lock_and_cache)
                {
                  if (lock_and_cache.first == false)
                    {
                      // Mark this entry as no longer being available for use.
                      lock_and_cache.first = true;
                      return lock_and_cache.second;
                    }
                }
            }
        }
      else
        {
          return get_destination_cache(scratch_data);
        }
    }

    template <int dim, int spacedim>
    static void
    return_source_cache_to_pool(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const GeneralDataStorage &              cache)
    {
      if (has_user_cache(scratch_data))
        {
          AD_SD_Functor_Cache &source_cache = get_source_cache(scratch_data);
          for (auto &lock_and_cache : source_cache.source_lock_and_cache)
            {
              if (&cache == &lock_and_cache.second)
                {
                  Assert(lock_and_cache.first == true,
                         ExcMessage("Cache entry was not locked upon return."));

                  // Mark this entry as being available for use.
                  lock_and_cache.first = false;
                  return;
                }
            }

          AssertThrow(false,
                      ExcMessage(
                        "Source cache not found in this data structure."));
        }
    }

    template <int dim, int spacedim>
    static GeneralDataStorage &
    get_destination_cache(MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      return scratch_data.get_general_data_storage();
    }

    template <int dim, int spacedim>
    static const GeneralDataStorage &
    get_destination_cache(
      const MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      return scratch_data.get_general_data_storage();
    }

    bool
    all_entries_unlocked() const
    {
      for (const auto &lock_and_cache : source_lock_and_cache)
        {
          if (lock_and_cache.first == true)
            return false;
        }

      return true;
    }

    std::size_t
    queue_length() const
    {
      return source_lock_and_cache.size();
    }

  private:
    // We need to be careful when a shared cache is used: We cannot evaluate
    // this operator in parallel; it must be done in a sequential fashion.
    Threads::Mutex mutex;

    using CacheWithLock = std::pair<bool, GeneralDataStorage>;
    std::vector<CacheWithLock> source_lock_and_cache;

    static std::string
    get_name_ad_sd_cache()
    {
      return Operators::internal::get_deal_II_prefix() + "AD_SD_Functor_Cache";
    }

    // static std::string
    // get_name_ad_sd_source_cache(const unsigned int entry)
    // {
    //   return get_name_ad_sd_cache + "_" + std::to_string(entry);
    // }

    template <int dim, int spacedim>
    static bool
    has_user_cache(MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      const GeneralDataStorage &scratch_cache =
        scratch_data.get_general_data_storage();
      return scratch_cache.stores_object_with_name(get_name_ad_sd_cache());
    }

    template <int dim, int spacedim>
    static bool
    has_user_cache(const MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      const GeneralDataStorage &scratch_cache =
        scratch_data.get_general_data_storage();
      return scratch_cache.stores_object_with_name(get_name_ad_sd_cache());
    }

    template <int dim, int spacedim>
    static AD_SD_Functor_Cache &
    get_source_cache(MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      Assert(has_user_cache(scratch_data),
             ExcMessage("No user cache exists in this scratch data object."));
      GeneralDataStorage &scratch_cache =
        scratch_data.get_general_data_storage();
      return scratch_cache.template get_object_with_name<AD_SD_Functor_Cache>(
        get_name_ad_sd_cache());
    }
  };
} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE


#endif // dealii_weakforms_ad_sd_functor_cache_h
