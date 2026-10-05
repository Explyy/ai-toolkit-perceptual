export type Point = { x: number; y: number };
export type Rectangle = { left: number; top: number; right: number; bottom: number };
export type SelectionTarget = { id: string; rect: Rectangle };
export type SelectionGesture = {
  scope: object;
  pointerId: number;
  start: Point;
  end: Point;
  targets: SelectionTarget[];
  excluded: 0 | 1;
};
export function selectionRectangle(start: Point, end: Point): Rectangle {
  return {
    left: Math.min(start.x, end.x),
    top: Math.min(start.y, end.y),
    right: Math.max(start.x, end.x),
    bottom: Math.max(start.y, end.y),
  };
}
export function selectionHits(gesture: SelectionGesture): string[] {
  const box = selectionRectangle(gesture.start, gesture.end);
  return gesture.targets
    .filter(
      ({ rect }) => rect.left < box.right && rect.right > box.left && rect.top < box.bottom && rect.bottom > box.top,
    )
    .map(x => x.id);
}
// A frozen inventory cannot acquire newly visible targets; a changed view cancels the gesture.
export function selectionCommit(gesture: SelectionGesture | null, scope: object, visibleIds: string[]): string[] {
  if (
    !gesture ||
    gesture.scope !== scope ||
    gesture.targets.length !== visibleIds.length ||
    gesture.targets.some((x, i) => x.id !== visibleIds[i]) ||
    Math.hypot(gesture.end.x - gesture.start.x, gesture.end.y - gesture.start.y) < 5
  )
    return [];
  return selectionHits(gesture);
}
