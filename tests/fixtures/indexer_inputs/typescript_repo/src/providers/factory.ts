import type { ProviderSettings } from "../settings";
import { AnthropicProvider } from "./anthropic";
import { Provider } from "./base";

export function createProvider(settings: ProviderSettings): Provider {
  return new AnthropicProvider(settings.apiKey);
}
