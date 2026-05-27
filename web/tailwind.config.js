/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "hsl(0 0% 100%)",
        foreground: "hsl(240 10% 3.9%)",
        muted: "hsl(240 4.8% 95.9%)",
        "muted-foreground": "hsl(240 3.8% 46.1%)",
        border: "hsl(240 5.9% 90%)",
        input: "hsl(240 5.9% 90%)",
        ring: "hsl(240 5.9% 10%)",
        primary: "hsl(240 5.9% 10%)",
        "primary-foreground": "hsl(0 0% 98%)",
        secondary: "hsl(240 4.8% 95.9%)",
        "secondary-foreground": "hsl(240 5.9% 10%)",
        accent: "hsl(240 4.8% 95.9%)",
        "accent-foreground": "hsl(240 5.9% 10%)",
        card: "hsl(0 0% 100%)",
        "card-foreground": "hsl(240 10% 3.9%)",
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "-apple-system", "sans-serif"],
        mono: ["'JetBrains Mono'", "ui-monospace", "monospace"],
      },
      borderRadius: {
        DEFAULT: "0.5rem",
      },
    },
  },
  plugins: [],
};
