import { Provider } from "./base";

export class AnthropicProvider extends Provider {
  private apiKey: string;

  constructor(apiKey: string) {
    super();
    this.apiKey = apiKey;
  }

  async complete(prompt: string): Promise<string> {
    return this.call(prompt);
  }

  private call(prompt: string): Promise<string> {
    return Promise.resolve(prompt);
  }
}
