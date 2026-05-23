#pragma once

#include <cstdlib>
#include <fstream>
#include <sstream>

#include <spdlog/spdlog.h>

#include <CGraph.h>

namespace app {

template <typename RegisterElementsFn>
int RunThreadedPipeline(int thread_num, RegisterElementsFn register_elements) {
	auto pipeline = CGraph::GPipelineFactory::create();

	auto st = register_elements(pipeline);
	if (st.isErr()) {
		spdlog::error("register failed: {}", st.getInfo());
		CGraph::GPipelineFactory::clear();
		return -1;
	}

	const char *dump_path = std::getenv("CGRAPH_DUMP_DOT_PATH");
	const bool dump_only = std::getenv("CGRAPH_DUMP_ONLY") != nullptr;
	if (dump_path && dump_path[0] != '\0') {
		std::ostringstream oss;
		st = pipeline->dump(oss);
		if (st.isErr()) {
			spdlog::error("dump cgraph failed: {}", st.getInfo());
			CGraph::GPipelineFactory::clear();
			return -1;
		}

		std::ofstream out(dump_path);
		if (!out.is_open()) {
			spdlog::error("open cgraph dump file failed: {}", dump_path);
			CGraph::GPipelineFactory::clear();
			return -1;
		}
		out << oss.str();
		spdlog::info("cgraph dumped to {}", dump_path);

		if (dump_only) {
			CGraph::GPipelineFactory::clear();
			return 0;
		}
	}

	CGraph::UThreadPoolConfig cfg;
	cfg.default_thread_size_ = thread_num;
	cfg.secondary_thread_size_ = 0;
	pipeline->setUniqueThreadPoolConfig(cfg);

	st = pipeline->process();
	if (st.isErr()) {
		spdlog::error("process exit: {}", st.getInfo());
	}

	CGraph::GPipelineFactory::clear();
	return st.isErr() ? -1 : 0;
}

}  // namespace app
