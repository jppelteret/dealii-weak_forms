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

#ifndef dealii_weakforms_ad_sd_functor_cache_h
#define dealii_weakforms_ad_sd_functor_cache_h

#include <deal.II/base/config.h>

#include <deal.II/algorithms/general_data_storage.h>

#include <deal.II/base/multithread_info.h>
#include <deal.II/base/thread_local_storage.h>
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
      : source_lock_and_cache(queue_length)
    {}

    AD_SD_Functor_Cache(const AD_SD_Functor_Cache &) = delete;
    AD_SD_Functor_Cache(AD_SD_Functor_Cache &&)      = delete;

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

      // Register the user cache in a central and always accessible place.
      AD_SD_Functor_Cache &user_cache =
        const_cast<AD_SD_Functor_Cache &>(*ad_sd_functor_cache);
      scratch_cache.add_unique_reference<AD_SD_Functor_Cache>(
        get_name_ad_sd_cache(), user_cache);
    }

    template <int dim, int spacedim>
    static void
    bind_user_cache_to_thread(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      if (has_user_cache(scratch_data) == false)
        return;

      // We must ensure that we do not try to evaluate from multiple threads
      // at once.
      AD_SD_Functor_Cache &user_cache = get_user_cache(scratch_data);

      // We use an infinite loop because we cannot be sure when the next
      // entry will become free for use.
      while (true)
        {
          for (auto &lock_and_cache : user_cache.source_lock_and_cache)
            {
              if (lock_and_cache.first.try_lock())
                {
                  // Now that we've marked this entry as no longer being
                  // available for use, and point the current thread towards
                  // its associated cache data.
                  GeneralDataStorage &data_storage =
                    get_data_storage(scratch_data);

#ifdef DEBUG
                  if (data_storage.stores_object_with_name(
                        get_name_active_data_storage()))
                    {
                      Assert(
                        data_storage.get_object_with_name<GeneralDataStorage *>(
                          get_name_active_data_storage()) == nullptr,
                        ExcMessage(
                          "Expected to find an uninitialised pointer to an active data storage object."));
                    }
#endif

                  data_storage.add_or_overwrite_copy<GeneralDataStorage *>(
                    get_name_active_data_storage(), &lock_and_cache.second);
                  return;
                }
            }
        }
    }

    template <int dim, int spacedim>
    static void
    unbind_user_cache_from_thread(
      MeshWorker::ScratchData<dim, spacedim> &scratch_data,
      const GeneralDataStorage &              cache)
    {
      if (has_user_cache(scratch_data) == false)
        return;

      AD_SD_Functor_Cache &user_cache = get_user_cache(scratch_data);
      for (auto &lock_and_cache : user_cache.source_lock_and_cache)
        {
          if (&cache == &lock_and_cache.second)
            {
              Assert(lock_and_cache.first.try_lock() == false,
                     ExcMessage("Cache entry was not locked upon return."));

              GeneralDataStorage &data_storage = get_data_storage(scratch_data);
              Assert(
                data_storage.stores_object_with_name(
                  get_name_active_data_storage()),
                ExcMessage(
                  "Expected to find a pointer to an active data storage object."));


              // Invalidate pointer in common storage and mark this entry as
              // being available for re-use.
              data_storage.add_or_overwrite_copy<GeneralDataStorage *>(
                get_name_active_data_storage(), nullptr);
              lock_and_cache.first.unlock();
              return;
            }
        }

      AssertThrow(false,
                  ExcMessage("Source cache not found in this data structure."));
    }

    template <int dim, int spacedim>
    static GeneralDataStorage &
    get_cache(MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      if (has_user_cache(scratch_data))
        {
          GeneralDataStorage &data_storage = get_data_storage(scratch_data);

          Assert(
            data_storage.stores_object_with_name(
              get_name_active_data_storage()),
            ExcMessage(
              "Expected to find a pointer to an active data storage object."));
          GeneralDataStorage *active_data_storage =
            data_storage.get_object_with_name<GeneralDataStorage *>(
              get_name_active_data_storage());

          Assert(
            active_data_storage != nullptr,
            ExcMessage(
              "Expected to find an initialised pointer to an active data storage object."));
          return *active_data_storage;
        }
      else
        {
          return get_data_storage(scratch_data);
        }
    }

    template <int dim, int spacedim>
    static const GeneralDataStorage &
    get_cache(const MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      if (has_user_cache(scratch_data))
        {
          // GeneralDataStorage does not have any non-const member functions.
          GeneralDataStorage &data_storage = get_data_storage(
            const_cast<MeshWorker::ScratchData<dim, spacedim> &>(scratch_data));

          Assert(
            data_storage.stores_object_with_name(
              get_name_active_data_storage()),
            ExcMessage(
              "Expected to find a pointer to an active data storage object."));

          const GeneralDataStorage *const active_data_storage =
            data_storage.get_object_with_name<GeneralDataStorage *>(
              get_name_active_data_storage());
          Assert(
            active_data_storage != nullptr,
            ExcMessage(
              "Expected to find an initialised pointer to an active data storage object."));

          return *active_data_storage;
        }
      else
        {
          return get_data_storage(scratch_data);
        }
    }

    std::size_t
    queue_length() const
    {
      return source_lock_and_cache.size();
    }

  private:
    // We need to be careful when a shared cache is used: We cannot evaluate
    // this operator in parallel; it must be done in a sequential fashion.
    using CacheWithLock = std::pair<std::mutex, GeneralDataStorage>;
    std::vector<CacheWithLock> source_lock_and_cache;

    static std::string
    get_name_ad_sd_cache()
    {
      return Utilities::get_deal_II_prefix() + "AD_SD_Functor_Cache";
    }

    static std::string
    get_name_active_data_storage()
    {
      return get_name_ad_sd_cache() + "_active_data_storage";
    }

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
    get_user_cache(MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      Assert(has_user_cache(scratch_data),
             ExcMessage("No user cache exists in this scratch data object."));
      GeneralDataStorage &scratch_cache =
        scratch_data.get_general_data_storage();
      return scratch_cache.template get_object_with_name<AD_SD_Functor_Cache>(
        get_name_ad_sd_cache());
    }

    template <int dim, int spacedim>
    static const AD_SD_Functor_Cache &
    get_user_cache(const MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      Assert(has_user_cache(scratch_data),
             ExcMessage("No user cache exists in this scratch data object."));
      const GeneralDataStorage &scratch_cache =
        scratch_data.get_general_data_storage();
      return scratch_cache.template get_object_with_name<AD_SD_Functor_Cache>(
        get_name_ad_sd_cache());
    }

    template <int dim, int spacedim>
    static GeneralDataStorage &
    get_data_storage(MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      return scratch_data.get_general_data_storage();
    }

    template <int dim, int spacedim>
    static const GeneralDataStorage &
    get_data_storage(const MeshWorker::ScratchData<dim, spacedim> &scratch_data)
    {
      return scratch_data.get_general_data_storage();
    }
  };
} // namespace WeakForms


WEAK_FORMS_NAMESPACE_CLOSE


#endif // dealii_weakforms_ad_sd_functor_cache_h
