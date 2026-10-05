// One page-local lane. Pending user intents take priority over new background work.
export class ActionLane {
  private occupied = false;
  private waiting: { valid: () => boolean; resolve: (release: (() => void) | null) => void }[] = [];
  private replay = new Map<string, Promise<any>>();
  get busy() {
    return this.occupied;
  }
  tryBackground(): (() => void) | null {
    if (this.occupied || this.waiting.length) return null;
    this.occupied = true;
    return this.release();
  }
  run<T>(key: string, valid: () => boolean, action: () => Promise<T>): Promise<T | undefined> {
    const previous = this.replay.get(key);
    if (previous) return previous;
    const result = (async () => {
      const release = await new Promise<(() => void) | null>(resolve => {
        this.waiting.push({ valid, resolve });
        this.drain();
      });
      if (!release) return;
      try {
        if (valid()) return await action();
      } finally {
        release();
      }
    })();
    this.replay.set(key, result);
    void result
      .finally(() => {
        if (this.replay.get(key) === result) this.replay.delete(key);
      })
      .catch(() => {});
    return result;
  }
  private release() {
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.occupied = false;
      this.drain();
    };
  }
  private drain() {
    while (!this.occupied && this.waiting.length) {
      const next = this.waiting.shift()!;
      if (!next.valid()) {
        next.resolve(null);
        continue;
      }
      this.occupied = true;
      next.resolve(this.release());
    }
  }
}
