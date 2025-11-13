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
  webpack: (config, { isServer }) => {
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
    config.resolve.extensionAlias = {
      '.js': ['.js', '.ts', '.tsx'],
      '.jsx': ['.jsx', '.tsx'],
    };
    config.resolve.alias = {
      ...(config.resolve.alias || {}),
      '@deck.gl/widgets': false,
      '@deck.gl/mapbox': false,
    };
    return config;
  },
};

export default nextConfig;
