#pragma once

#include <iostream>

#include <CGraph.h>

namespace app {

template <typename RegisterElementsFn>
int RunThreadedPipeline(int thread_num, RegisterElementsFn register_elements) {
	auto pipeline = CGraph::GPipelineFactory::create();

	auto st = register_elements(pipeline);
	if (st.isErr()) {
		std::cerr << "register failed: " << st.getInfo() << std::endl;
		CGraph::GPipelineFactory::clear();
		return -1;
	}

	CGraph::UThreadPoolConfig cfg;
	cfg.default_thread_size_ = thread_num;
	cfg.secondary_thread_size_ = 0;
	pipeline->setUniqueThreadPoolConfig(cfg);

	st = pipeline->process();
	if (st.isErr()) {
		std::cerr << "process exit: " << st.getInfo() << std::endl;
	}

	CGraph::GPipelineFactory::clear();
	return st.isErr() ? -1 : 0;
}

}  // namespace app
