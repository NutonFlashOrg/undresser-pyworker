---
name: senior-dev-style-frontend
description: TypeScript/React code quality bar for the frontend repo. Strict TS, no any, small composable components, state machines for multi-step flows, mobile-first, Russian copy in registry.
---

# Frontend Code Style — TypeScript / React

## TypeScript

- Strict mode is on (`"strict": true` in tsconfig). Do not disable it.
- No `any`. Use `unknown` and narrow it, or define a proper type.
- No `@ts-ignore` or `@ts-expect-error` without a comment explaining the constraint.
- Prefer `interface` for object shapes that may be extended; `type` for unions and intersections.

## React components

- Components should be small and do one thing. If a component file exceeds ~200 lines, consider splitting.
- Use functional components and hooks. No class components.
- Co-locate styles and logic with the component unless the pattern is shared.
- Avoid prop drilling beyond two levels — use context or a state solution.
- State machines (e.g. `useReducer` with an explicit state enum) for multi-step flows: generation progress, dataset upload, character creation wizard.

## Accessibility

- Every user-facing interactive element (buttons, links, inputs) must have a meaningful label or `aria-label`.
- Do not rely on color alone to convey meaning.
- Keyboard navigation must work for modals and drawers.

## Mobile-first

- Telegram WebApp viewport. Design for 390px width first.
- Use relative units and flex/grid layouts. No fixed pixel widths for content areas.
- Test touch targets: minimum 44px × 44px.

## Copy and i18n

- Russian-language copy lives in the existing copy registry. Do not inline magic strings in JSX.
- Refer to the `frontend-ru-copy` skill and the existing copy registry before adding new strings.
- Do not add new i18n keys without following the existing registration pattern.

## Lint and type check

- `npm run lint` — ESLint with zero warnings allowed
- `tsc --noEmit` — type check without emitting (runs as part of `npm run build`)
- `npm test` — Vitest unit tests

## What not to do

- Do not use `console.log` in committed code.
- Do not import from `../../../..` — use path aliases configured in tsconfig.
- Do not add inline `style={{}}` for layout; use CSS classes.
- Do not merge a component that breaks an existing page — run the dev server and verify the golden path.
