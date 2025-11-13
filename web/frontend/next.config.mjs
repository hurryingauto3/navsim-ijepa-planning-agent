import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Exclude onnxruntime-web from optimization to avoid minification issues
  experimental: {
    outputFileTracingExcludes: {
      '*': [
        'node_modules/onnxruntime-web/**/*.mjs',
        'node_modules/onnxruntime-web/**/ort.bundle*.mjs',
        'node_modules/onnxruntime-web/**/ort.node*.mjs',
      ],
    },
  },
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
    
    // Configure webpack to handle .mjs files from onnxruntime-web
    // These files contain ES module syntax (import.meta) that minifiers can't handle
    // Configure module parser to preserve ES module syntax
    config.module.parser = config.module.parser || {};
    config.module.parser.javascript = config.module.parser.javascript || {};
    
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
    
    // Configure optimization to exclude onnxruntime-web from minification
    // These files contain ES module syntax (import.meta) that Terser can't handle
    if (!isServer) {
      // Ensure optimization object exists
      config.optimization = config.optimization || {};
      config.optimization.minimizer = config.optimization.minimizer || [];
      
      // Find and update the Terser plugin configuration
      const terserPluginIndex = config.optimization.minimizer.findIndex(
        (plugin) => {
          // Check if it's a Terser plugin (Next.js uses SwcMinify in newer versions, but may fall back to Terser)
          return plugin && (
            plugin.constructor.name === 'TerserPlugin' ||
            (plugin.options && plugin.options.terserOptions)
          );
        }
      );
      
      if (terserPluginIndex !== -1) {
        const terserPlugin = config.optimization.minimizer[terserPluginIndex];
        
        // Update the exclude pattern to exclude onnxruntime-web files
        const originalExclude = terserPlugin.options?.exclude;
        terserPlugin.options = terserPlugin.options || {};
        
        // Configure Terser to handle ES modules and exclude onnxruntime-web
        terserPlugin.options.exclude = (file) => {
          // Exclude onnxruntime-web files from minification
          if (file.includes('onnxruntime-web') || 
              file.includes('ort.bundle') || 
              file.includes('ort.node') ||
              file.includes('.mjs')) {
            return true;
          }
          // Apply original exclude if it exists
          if (typeof originalExclude === 'function') {
            return originalExclude(file);
          }
          if (originalExclude instanceof RegExp) {
            return originalExclude.test(file);
          }
          return false;
        };
        
        // Configure Terser to handle ES modules properly
        terserPlugin.options.terserOptions = {
          ...(terserPlugin.options.terserOptions || {}),
          module: true,
          ecma: 2020,
          parse: {
            ecma: 2020,
          },
          compress: {
            ...(terserPlugin.options.terserOptions?.compress || {}),
            ecma: 2020,
          },
          format: {
            ...(terserPlugin.options.terserOptions?.format || {}),
            ecma: 2020,
          },
        };
      }
      
      // Configure webpack to skip minification for onnxruntime-web chunks
      // Use splitChunks to isolate onnxruntime-web into separate chunks that won't be minified
      if (!config.optimization.splitChunks) {
        config.optimization.splitChunks = {
          chunks: 'all',
          cacheGroups: {
            default: false,
            vendors: false,
            onnxruntime: {
              name: 'onnxruntime',
              test: /[\\/]node_modules[\\/]onnxruntime-web[\\/]/,
              chunks: 'all',
              priority: 20,
              enforce: true,
            },
          },
        };
      }
    }
    
    return config;
  },
};

export default nextConfig;
