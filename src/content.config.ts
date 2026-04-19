import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

const posts = defineCollection({
  loader: glob({ pattern: "**/*.md", base: "./src/content/posts" }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    lang: z.enum(["en", "ko"]),
    categories: z.array(z.string()),
    tags: z.array(z.string()),
    series: z.string().optional(),
    part: z.number().optional(),
    excerpt: z.string().optional(),
    alt_lang: z.string().optional(),
    next_title: z.string().optional(),
    next_desc: z.string().optional(),
    next_status: z.enum(["draft", "published"]).optional(),
    source_url: z.string().optional(),
    source_label: z.string().optional(),
  }),
});

export const collections = { posts };
