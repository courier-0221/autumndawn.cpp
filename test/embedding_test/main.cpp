#define OBX_CPP_FILE  // 让 objectbox.hpp 中的实现在本文件"materialize"

#include <iostream>

#include "config.hpp"
#include "embedding_test_app.hpp"
#include "objectbox-model.h"
#include "objectbox.hpp"
#include "siliconflow_client.hpp"

// export LD_LIBRARY_PATH=path/third_party/objectbox/lib:$LD_LIBRARY_PATH
// ./embedding_test

int main() {
    std::cout << "** ObjectBox vector search + SiliconFlow embedding demo **" << std::endl;

    if (!obx_has_feature(OBXFeature_VectorSearch)) {
        std::cerr << "Vector search is not supported by this ObjectBox build." << std::endl;
        return 1;
    }

    AppConfig config;
    try {
        config = loadConfig("emb_config.json");
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    SiliconFlowEmbeddingClient client(config.baseUrl, config.apiKey, config.model);

    obx::Options options(create_obx_model());
    options.directory("objectbox-db");
    obx::Store store(options);

    EmbeddingTestApp app(store, client);
    return app.run();
}
