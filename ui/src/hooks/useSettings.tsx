'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export interface Settings {
  HF_TOKEN: string;
  HF_TOKEN_CONFIGURED: boolean;
  CLEAR_HF_TOKEN: boolean;
  TRAINING_FOLDER: string;
  DATASETS_FOLDER: string;
  MODELS_PATH: string;
}

export default function useSettings() {
  const [settings, setSettings] = useState({
    HF_TOKEN: '',
    HF_TOKEN_CONFIGURED: false,
    CLEAR_HF_TOKEN: false,
    TRAINING_FOLDER: '',
    DATASETS_FOLDER: '',
    MODELS_PATH: '',
  });
  const [isSettingsLoaded, setIsLoaded] = useState(false);
  useEffect(() => {
    apiClient
      .get('/api/settings')
      .then(res => res.data)
      .then(data => {
        setSettings({
          HF_TOKEN: '',
          HF_TOKEN_CONFIGURED: data.HF_TOKEN_CONFIGURED === true,
          CLEAR_HF_TOKEN: false,
          TRAINING_FOLDER: data.TRAINING_FOLDER || '',
          DATASETS_FOLDER: data.DATASETS_FOLDER || '',
          MODELS_PATH: data.MODELS_PATH || '',
        });
        setIsLoaded(true);
      })
      .catch(error => console.error('Error fetching settings:', error));
  }, []);

  return { settings, setSettings, isSettingsLoaded };
}
