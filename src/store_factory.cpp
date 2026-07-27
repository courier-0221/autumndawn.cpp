#include "rag/store_factory.hpp"

#include <type_traits>
#include <variant>

#include "storage/objectbox/objectbox_store.hpp"

namespace autumndawn::rag {

std::shared_ptr<storage::IVectorStore> createVectorStore(const StorageBackendConfig& cfg,
                                                          std::size_t embeddingDim) {
    return std::visit(
        [embeddingDim](const auto& c) -> std::shared_ptr<storage::IVectorStore> {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, ObjectBoxStorageConfig>) {
                return std::make_shared<storage::ObjectBoxVectorStore>(c.directory,
                                                                       embeddingDim);
            }
        },
        cfg);
}

}  // namespace autumndawn::rag
