declare module 'tkyodrift' {
  /**
   * Analyzes drift for a single text input and saves embeddings and scalar metrics.
   * @param text - The input or output string to analyze.
   * @param ioType - The type of text (e.g. "input", "output", or custom).
   * @returns A Promise that resolves when the drift analysis is complete.
   */
  export default function tkyoDrift(
    text: string,
    ioType: string
  ): Promise<void>;
}
