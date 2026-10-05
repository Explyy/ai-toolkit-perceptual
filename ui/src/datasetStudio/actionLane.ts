type LoadRequest = { owner: object; sequence: number; errorRevision: number };

// A read may clear its previous load failure, but never a later action/save error.
export class LoadErrorOwner {
  private sequence = 0;
  private errorRevision = 0;
  private loadError: { owner: object; revision: number } | null = null;
  action() {
    this.errorRevision++;
    this.loadError = null;
  }
  begin(owner: object): LoadRequest {
    return { owner, sequence: ++this.sequence, errorRevision: this.errorRevision };
  }
  current(request: LoadRequest, owner: object) {
    return request.owner === owner && request.sequence === this.sequence;
  }
  accepts(request: LoadRequest, owner: object, receivedRevision: number, currentRevision?: number) {
    return (
      this.current(request, owner) &&
      Number.isSafeInteger(receivedRevision) &&
      (currentRevision === undefined || receivedRevision >= currentRevision)
    );
  }
  failed(request: LoadRequest, owner: object) {
    if (!this.current(request, owner) || request.errorRevision !== this.errorRevision) return false;
    this.loadError = { owner, revision: ++this.errorRevision };
    return true;
  }
  succeeded(request: LoadRequest, owner: object) {
    if (
      !this.current(request, owner) ||
      request.errorRevision !== this.errorRevision ||
      this.loadError?.owner !== owner ||
      this.loadError.revision !== this.errorRevision
    )
      return false;
    this.action();
    return true;
  }
}

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
