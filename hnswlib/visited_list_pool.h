// ==================================================================
// hnswlib/visited_list_pool.h
// ==================================================================
// Documentation for the VisitedList and VisitedListPool
// classes. These utilities are used internally by HNSW to mark
// nodes that have been visited during a search, preventing
// redundant processing without clearing the entire array each time.
//
// The technique: each search gets a unique "tag" (an incrementing
// counter). The visited array stores the tag of the search that last
// visited each node. When the tag overflows, the array is cleared.
// This avoids O(N) memory initialization per search.
//
// Author:   hnswlib contributors
// License:  Apache 2.0
// ==================================================================

#pragma once

#include <mutex>
#include <string.h>
#include <deque>

namespace hnswlib {

// ------------------------------------------------------------------
// Type used for visited tags (unsigned short, up to 65535 unique tags)
// ------------------------------------------------------------------
typedef unsigned short int vl_type;

// ------------------------------------------------------------------
// VisitedList: per‑thread or per‑search visitation tracker
// ------------------------------------------------------------------

/**
 * @brief Tracks which nodes have been visited during a single search.
 *
 * Each VisitedList contains an array `mass` of size `numelements`,
 * where each element stores a tag (curV). Initially, all tags are 0.
 * Before a search, we increment curV. When we visit a node, we set
 * mass[node] = curV. To check if a node was visited in this search,
 * we compare mass[node] == curV. This is much faster than resetting
 * the entire array after each search.
 *
 * When curV reaches its maximum value (65535 for unsigned short),
 * the next increment would wrap to 0, causing collisions with the
 * initial state. Therefore, reset() checks for overflow and, if needed,
 * clears the whole array with memset and increments curV to 1.
 */
class VisitedList {
 public:
    vl_type curV;           ///< Current tag for this list (incremented per use)
    vl_type *mass;          ///< Array of tags, one per node (index = internal ID)
    unsigned int numelements; ///< Number of nodes in the index (max size)

    /**
     * @brief Constructs a VisitedList for an index of given size.
     * @param numelements1 Number of nodes (should match max_elements_ in HNSW).
     */
    VisitedList(int numelements1) {
        curV = -1;                // will be incremented to 0 on first reset
        numelements = numelements1;
        mass = new vl_type[numelements];
    }

    /**
     * @brief Prepares the list for a new search.
     *
     * Increments curV. If curV would wrap to 0, we clear the entire
     * mass array and set curV = 1, because a tag of 0 is reserved for
     * "never visited" (initial state after clear).
     */
    void reset() {
        curV++;
        if (curV == 0) {          // overflow (wrapped from 65535 to 0)
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;               // now curV = 1 (safe, not 0)
        }
    }

    ~VisitedList() { delete[] mass; }
};

// ------------------------------------------------------------------
// VisitedListPool: thread‑safe pool of VisitedList objects
// ------------------------------------------------------------------

/**
 * @brief Manages a pool of VisitedList objects for concurrent searches.
 *
 * Since multiple threads may perform searches simultaneously, each
 * search requires its own VisitedList to avoid tag collisions.
 * This pool maintains a collection of pre‑allocated VisitedList
 * instances and reuses them. If the pool is exhausted, a new one is
 * created on the fly. When a search finishes, the list is returned
 * to the pool for future reuse.
 *
 * The pool is thread‑safe via a mutex. The number of initial lists
 * should be at least the expected concurrency level.
 */
class VisitedListPool {
    std::deque<VisitedList *> pool;   ///< Queue of available VisitedList objects
    std::mutex poolguard;             ///< Mutex for thread‑safe access to the pool
    int numelements;                  ///< Size of each VisitedList (number of index nodes)

 public:
    /**
     * @brief Constructs a pool with an initial number of lists.
     * @param initmaxpools Initial number of VisitedList objects to create.
     * @param numelements1 Number of nodes in the index (max_elements_).
     */
    VisitedListPool(int initmaxpools, int numelements1) {
        numelements = numelements1;
        for (int i = 0; i < initmaxpools; i++)
            pool.push_front(new VisitedList(numelements));
    }

    /**
     * @brief Obtains a VisitedList for use in a search.
     *
     * If the pool has an available list, it is taken from the front.
     * Otherwise, a new list is allocated. The list is then reset
     * (tag incremented) and returned.
     *
     * @return Pointer to a VisitedList ready for a new search.
     */
    VisitedList *getFreeVisitedList() {
        VisitedList *rez;
        {
            std::unique_lock <std::mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            } else {
                rez = new VisitedList(numelements);
            }
        }
        rez->reset();   // prepare for the new search
        return rez;
    }

    /**
     * @brief Returns a VisitedList to the pool after a search finishes.
     * @param vl The VisitedList to release (must have been obtained from this pool).
     */
    void releaseVisitedList(VisitedList *vl) {
        std::unique_lock <std::mutex> lock(poolguard);
        pool.push_front(vl);
    }

    /**
     * @brief Destructor: deletes all VisitedList objects in the pool.
     */
    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList *rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }
};

}  // namespace hnswlib
