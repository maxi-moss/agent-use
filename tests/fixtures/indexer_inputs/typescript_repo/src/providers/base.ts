export abstract class Provider {
  abstract complete(prompt: string): Promise<string>;
}
