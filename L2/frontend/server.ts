
import { CopilotRuntime, createCopilotEndpoint } from "@copilotkit/runtime/v2";
import { LangGraphHttpAgent } from "@copilotkit/runtime/langgraph";
import { serve } from "@hono/node-server";

const langGraphAgent = new LangGraphHttpAgent({
  url: process.env.LANGGRAPH_DEPLOYMENT_URL || "http://localhost:8002",
});

const runtime = new CopilotRuntime({
  agents: {
    default: langGraphAgent,
  },
});

const app = createCopilotEndpoint({
  runtime,
  basePath: "/api/copilotkit",
});

serve({ fetch: app.fetch, port: 4002 }, () => {
  console.log("CopilotKit API server running at http://localhost:4002");
});
