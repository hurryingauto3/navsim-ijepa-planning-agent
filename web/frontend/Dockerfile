FROM node:20-alpine

WORKDIR /app

COPY package.json ./
COPY pnpm-lock.yaml* ./
RUN corepack enable \
    && if [ -f pnpm-lock.yaml ]; then \
         pnpm install --frozen-lockfile; \
       else \
         pnpm install; \
       fi

COPY . .

EXPOSE 3000

CMD ["pnpm", "dev", "-p", "3000"]
