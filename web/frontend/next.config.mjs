import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  transpilePackages: [
    '@deck.gl/core',
    '@deck.gl/layers',
    '@deck.gl/react',
    '@deck.gl/aggregation-layers',
    '@deck.gl/geo-layers',
    '@deck.gl/mesh-layers',
    '@luma.gl/core',
    '@luma.gl/constants',
    '@luma.gl/webgl',
    '@math.gl/core',
    '@math.gl/web-mercator',
    '@loaders.gl/core',
    '@loaders.gl/gltf',
    '@loaders.gl/images',
    '@loaders.gl/loader-utils',
    '@loaders.gl/math',
    '@loaders.gl/schema',
    '@loaders.gl/tables',
    '@loaders.gl/terrain',
    '@loaders.gl/tiles',
    '@loaders.gl/worker-utils',
  ],
  webpack: (config, { isServer, webpack }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
      };
    }
    config.module = config.module || {};
    config.module.rules = config.module.rules || [];
    
    // Replace Node.js-specific files from onnxruntime-web with empty module
    // These files contain Node.js-specific code that shouldn't be bundled for browser
    config.plugins = config.plugins || [];
    config.plugins.push(
      new webpack.NormalModuleReplacementPlugin(
        /\.node\.mjs$/,
        path.resolve(__dirname, 'webpack-empty-module.js')
      )
    );
    
    // Also ignore via IgnorePlugin as a fallback
    config.plugins.push(
      new webpack.IgnorePlugin({
        checkResource(resource) {
          // Ignore Node.js-specific files from onnxruntime-web
          if (resource.includes('ort.node') || resource.includes('.node.mjs')) {
            return true;
          }
          return false;
        },
      })
    );
    
    // Configure externals for server-side builds to exclude Node.js files
    if (isServer) {
      config.externals = config.externals || [];
      config.externals.push({
        'onnxruntime-node': 'commonjs onnxruntime-node',
      });
    }
    
    // Ignore onnxruntime-web's Node.js-specific imports during build
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      '@deck.gl/widgets': false,
      '@deck.gl/mapbox': false,
      'onnxruntime-node': false,
    };
    
    config.resolve.extensionAlias = {
      '.js': ['.js', '.ts', '.tsx'],
      '.jsx': ['.jsx', '.tsx'],
    };
    
    return config;
  },
};

export default nextConfig;
