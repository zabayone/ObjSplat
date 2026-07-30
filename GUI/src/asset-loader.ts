import { getInputFormat, ReadFileSystem } from '@playcanvas/splat-transform';
import { AppBase, Asset, GSplatResource, Vec3 } from 'playcanvas';

import { Events } from './events';
import { loadGSplatData, validateGSplatData } from './io';
import { Splat } from './splat';

// Approximate CPU-side budget before creating GSplatResource (GPU upload can exceed this).
// Disabled by default to allow large datasets to be loaded in the GUI.
const MAX_ESTIMATED_GSPLAT_BYTES = Infinity; // set to a number to re-enable safety check

const formatBytes = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
};

const estimateGsplatBytes = (gsplatData: any) => {
    const vertex = gsplatData?.getElement?.('vertex');
    if (!vertex?.properties?.length || !gsplatData?.numSplats) {
        return 0;
    }
    return vertex.properties.reduce((sum: number, prop: any) => {
        const b = Number(prop?.byteSize) || 0;
        return sum + b * gsplatData.numSplats;
    }, 0);
};

const getOrientation = (filename: string) => {
    switch (getInputFormat(filename)) {
        case 'spz':
            return new Vec3(0, 0, 0);
        case 'lcc':
            return new Vec3(90, 0, 180);
        default:
            return new Vec3(0, 0, 180);
    }
};

// handles loading gsplat assets using splat-transform
class AssetLoader {
    app: AppBase;
    events: Events;

    constructor(app: AppBase, events: Events) {
        this.app = app;
        this.events = events;
    }

    async load(filename: string, fileSystem: ReadFileSystem, animationFrame?: boolean, skipReorder?: boolean) {
        if (!animationFrame) {
            this.events.fire('startSpinner');
        }

        try {
            // Skip reordering for animation frames (speed) or when explicitly requested (already ordered)
            const gsplatData = await loadGSplatData(filename, fileSystem, skipReorder || animationFrame);
            validateGSplatData(gsplatData);

            const estimatedBytes = estimateGsplatBytes(gsplatData);
            if (estimatedBytes > MAX_ESTIMATED_GSPLAT_BYTES) {
                throw new Error(
                    `Dataset too large for interactive viewer safety (${formatBytes(estimatedBytes)} estimated raw data). ` +
                    'Please convert/compress or decimate before loading (for example: compressed PLY, reduced SH bands, or fewer splats).'
                );
            }

            const asset = new Asset(filename, 'gsplat', { url: `local-asset-${Date.now()}`, filename });
            this.app.assets.add(asset);
            asset.resource = new GSplatResource(this.app.graphicsDevice, gsplatData);

            return new Splat(asset, getOrientation(filename));
        } finally {
            if (!animationFrame) {
                this.events.fire('stopSpinner');
            }
        }
    }
}

export { AssetLoader };
