import type { Provider } from "./providers/base";
import { createProvider } from "./providers/factory";
import type { ProviderSettings } from "./settings";

/** Sends messages through the configured provider. */
export class ChatService {
  private provider: Provider;

  constructor(settings: ProviderSettings) {
    this.provider = createProvider(settings);
  }

  send = async (message: string): Promise<string> => {
    return this.provider.complete(message);
  };
}

export const handleChat = async (service: ChatService, message: string) => {
  return service.send(message);
};
