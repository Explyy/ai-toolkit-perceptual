import { captionerTypes } from '@/helpers/captionOptions';
import { ensure, text } from './domain';
export const captionModels = captionerTypes
  .filter(x => x.group.includes('image'))
  .flatMap(x =>
    (x.name_or_path_options ?? []).map(m => ({
      key: JSON.stringify([x.name, String(m.value)]),
      captioner: x.name,
      model: String(m.value),
      label: m.label,
      group: x.label,
    })),
  );
export const defaultCaptionModel =
  captionModels.find(x => x.captioner === 'Qwen3VLCaptioner' && x.model === 'Qwen/Qwen3-VL-2B-Instruct') ??
  captionModels[0];
export type CaptionPreferences = { key: string; instructions: string };
export function preferences(input: any): CaptionPreferences {
  ensure(input && typeof input === 'object', 'Scegli un modello locale');
  ensure(
    captionModels.some(x => x.key === input.key),
    'Modello non presente nel catalogo locale',
  );
  return { key: input.key, instructions: text(input.instructions, 8000) };
}
