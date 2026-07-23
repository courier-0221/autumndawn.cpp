#define OBX_CPP_FILE
#include "task.obx.hpp"
#include "objectbox-model.h"
#include <iostream>

// export LD_LIBRARY_PATH=xxx/third_party/objectbox/lib:$LD_LIBRARY_PATH
// ./obxdemo

static obx::Options makeOptions() {
    obx::Options options;
    options.model(create_obx_model());
    options.directory("objectbox-db");
    return options;
}

/// 第一阶段：写入数据，返回写入的 id 列表
static std::vector<obx_id> writeData() {
    obx::Store store(makeOptions());
    obx::Box<Task> box(store);

    std::vector<std::string> texts = {
        "Buy milk",
        "Walk the dog",
        "Learn ObjectBox",
    };

    std::vector<obx_id> ids;
    for (const auto& text : texts) {
        Task task;
        task.id = 0;  // 0 表示新插入，由 ObjectBox 自动分配 ID
        task.text = text;
        obx_id id = box.put(task);
        ids.push_back(id);
        std::cout << "[write] put id=" << id << "  text=\"" << text << "\"\n";
    }

    std::cout << "[write] store closed\n\n";
    return ids;  // store 在此析构，数据库关闭
}

/// 第二阶段：重新打开数据库，按 id 读取数据
static void readData(const std::vector<obx_id>& ids) {
    obx::Store store(makeOptions());
    obx::Box<Task> box(store);

    std::cout << "[read] reopened store, reading " << ids.size() << " records:\n";
    for (obx_id id : ids) {
        auto task = box.get(id);
        if (task) {
            std::cout << "[read] id=" << id << "  text=\"" << task->text << "\"\n";
        } else {
            std::cout << "[read] id=" << id << "  not found\n";
        }
    }

    // 读取全部数据
    auto all = box.getAll();
    std::cout << "\n[read] getAll() returned " << all.size() << " records total\n";
    for (const auto& t : all) {
        std::cout << "  id=" << t->id << "  text=\"" << t->text << "\"\n";
    }
}

int main() {
    // 第一阶段：写入
    auto ids = writeData();

    // 第二阶段：重新加载数据库并读取
    readData(ids);

    return 0;
}