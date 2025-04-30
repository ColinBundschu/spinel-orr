module.exports = {
  root: true,
  env: {
    browser: true,
    es2023: true,
  },
  extends: [
    'plugin:@typescript-eslint/recommended',
    'airbnb'
  ],
  ignorePatterns: ['dist', '.eslintrc.cjs', 'vite.config.ts'],
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2023,
    sourceType: 'module',
    ecmaFeatures: {
      jsx: true,
    },
    project: './tsconfig.json',
  },
  plugins: ['react', '@typescript-eslint'],
  rules: {
    'camelcase': 'off',
    'lines-between-class-members': ['error', 'always', { 'exceptAfterSingleLine': true }],
    'max-len': ['error', { 'code': 120 }],
    'jsx-a11y/label-has-associated-control': ['error', {
      'required': {
        'some': ['nesting', 'id']
      }
    }],
  },
  settings: {
    react: {
      version: 'detect',
    },
  },
  overrides: [
    {
      files: ['*.tsx'],
      rules: {
        'react/react-in-jsx-scope': 'off',
        'react/jsx-filename-extension': [2, { 'extensions': ['.tsx'] }],
      },
    },
    {
      files: ['*.ts', '*.tsx'],
      rules: {
        'no-undef': 'off', // Fixes "'JSX' is not defined." https://typescript-eslint.io/linting/troubleshooting/#i-get-errors-from-the-no-undef-rule-about-global-variables-not-being-defined-even-though-there-are-no-typescript-errors
        'no-unused-vars': 'off',
        '@typescript-eslint/no-unused-vars': 'error',
        'no-shadow': 'off',
        '@typescript-eslint/no-shadow': 'error',
      }
    },
  ],
};
