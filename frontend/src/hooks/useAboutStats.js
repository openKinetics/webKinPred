import { useEffect, useState } from 'react';
import apiClient from '../components/appClient';
import { ABOUT_STATS_STORAGE_KEY, isFreshAboutStats, readAboutStats } from '../utils/aboutStatsCache.mjs';

export default function useAboutStats() {
  const [stats, setStats] = useState(() => {
    try { return readAboutStats(window.localStorage); } catch { return null; }
  });

  useEffect(() => {
    let disposed = false;
    let controller = null;
    let lastAttempt = 0;
    let lastSuccess = 0;
    const refresh = async () => {
      if (disposed || controller || Date.now() - lastAttempt < 60_000) return;
      controller = new AbortController();
      lastAttempt = Date.now();
      try {
        const { data } = await apiClient.get('/about-stats/', { signal: controller.signal });
        if (disposed || !isFreshAboutStats(data)) return;
        lastSuccess = Date.now();
        setStats(data);
        try { window.localStorage.setItem(ABOUT_STATS_STORAGE_KEY, JSON.stringify(data)); } catch { /* optional */ }
      } catch {
        if (!disposed) setStats(previous => isFreshAboutStats(previous) ? previous : null);
      } finally {
        controller = null;
      }
    };
    const onVisible = () => { if (!document.hidden) refresh(); };
    refresh(); // Revalidate on every visit, even when a local preview was available.
    const timer = window.setInterval(() => {
      setStats(previous => isFreshAboutStats(previous) ? previous : null);
      if (!document.hidden && Date.now() - lastSuccess >= 5 * 60_000) refresh();
    }, 60_000);
    window.addEventListener('focus', onVisible);
    document.addEventListener('visibilitychange', onVisible);
    return () => {
      disposed = true;
      window.clearInterval(timer);
      window.removeEventListener('focus', onVisible);
      document.removeEventListener('visibilitychange', onVisible);
      controller?.abort();
    };
  }, []);
  return stats;
}
