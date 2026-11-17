# TypeScript conventions
- `strict` TypeScript; no `any` without a comment saying why. Prefer `unknown` plus narrowing.
- pnpm manages dependencies; add them with `pnpm add`, never edit the lockfile by hand.
- Components are small and typed; state lives as low in the tree as it can, and data fetching is separate from presentation.
- Validate external data (API responses, form input) at the boundary with a schema library such as zod.
- Test logic with vitest and user-visible behaviour with Testing Library; don't test layout or styling.
- ESLint and Prettier run in the gates; don't fight them.
