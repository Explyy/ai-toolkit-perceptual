import { NextResponse } from 'next/server';
import prisma from '@/server/prisma';
import { defaultTrainFolder, defaultDatasetsFolder, defaultModelsFolder } from '@/paths';
import { flushCache } from '@/server/settings';
import { sameOrigin } from '@/datasetStudio/http';

export async function GET() {
  try {
    const settings = await prisma.settings.findMany();
    const settingsObject = settings.reduce((acc: any, setting) => {
      acc[setting.key] = setting.value;
      return acc;
    }, {});
    // if TRAINING_FOLDER is not set, use default
    if (!settingsObject.TRAINING_FOLDER || settingsObject.TRAINING_FOLDER === '') {
      settingsObject.TRAINING_FOLDER = defaultTrainFolder;
    }
    // if DATASETS_FOLDER is not set, use default
    if (!settingsObject.DATASETS_FOLDER || settingsObject.DATASETS_FOLDER === '') {
      settingsObject.DATASETS_FOLDER = defaultDatasetsFolder;
    }
    // MODELS_PATH from the env file always takes precedence over the setting
    if (process.env.MODELS_PATH && process.env.MODELS_PATH.trim() !== '') {
      settingsObject.MODELS_PATH = process.env.MODELS_PATH;
    } else if (!settingsObject.MODELS_PATH || settingsObject.MODELS_PATH === '') {
      // if MODELS_PATH is not set, use default
      settingsObject.MODELS_PATH = defaultModelsFolder;
    }
    settingsObject.HF_TOKEN_CONFIGURED = !!settingsObject.HF_TOKEN;
    settingsObject.HF_TOKEN = '';
    if (process.env.DATASET_STUDIO_DATASETS_ROOT)
      settingsObject.DATASETS_FOLDER = process.env.DATASET_STUDIO_DATASETS_ROOT;
    if (process.env.DATASET_STUDIO_TRAINING_ROOT)
      settingsObject.TRAINING_FOLDER = process.env.DATASET_STUDIO_TRAINING_ROOT;
    return NextResponse.json(settingsObject);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch settings' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    sameOrigin(request);
    const body = await request.json();
    const { HF_TOKEN, CLEAR_HF_TOKEN, TRAINING_FOLDER, DATASETS_FOLDER, MODELS_PATH } = body;

    // Upsert both settings
    await Promise.all([
      ...(CLEAR_HF_TOKEN || HF_TOKEN
        ? [
            prisma.settings.upsert({
              where: { key: 'HF_TOKEN' },
              update: { value: CLEAR_HF_TOKEN ? '' : HF_TOKEN },
              create: { key: 'HF_TOKEN', value: CLEAR_HF_TOKEN ? '' : HF_TOKEN },
            }),
          ]
        : []),
      prisma.settings.upsert({
        where: { key: 'TRAINING_FOLDER' },
        update: { value: TRAINING_FOLDER },
        create: { key: 'TRAINING_FOLDER', value: TRAINING_FOLDER },
      }),
      prisma.settings.upsert({
        where: { key: 'DATASETS_FOLDER' },
        update: { value: DATASETS_FOLDER },
        create: { key: 'DATASETS_FOLDER', value: DATASETS_FOLDER },
      }),
      prisma.settings.upsert({
        where: { key: 'MODELS_PATH' },
        update: { value: MODELS_PATH },
        create: { key: 'MODELS_PATH', value: MODELS_PATH },
      }),
    ]);

    flushCache();

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to update settings' }, { status: 500 });
  }
}
