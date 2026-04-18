# frozen_string_literal: true

source "https://rubygems.org"

gem "jekyll-theme-chirpy", "~> 7.1"

# Plugins referenced by _config.yml — declared explicitly to guarantee
# installation. Redundant with theme gemspec dependencies, but the
# theme gem's runtime deps do not always propagate cleanly on CI.
gem "jekyll-paginate", "~> 1.1"
gem "jekyll-redirect-from", "~> 0.16"
gem "jekyll-seo-tag", "~> 2.8"
gem "jekyll-sitemap", "~> 1.4"
gem "jekyll-archives", "~> 2.2"

group :test do
  gem "html-proofer", "~> 5.0"
end

# Windows and JRuby do not include zoneinfo files
platforms :mingw, :x64_mingw, :mswin, :jruby do
  gem "tzinfo", ">= 1", "< 3"
  gem "tzinfo-data"
end

# Performance-booster for watching directories on Windows
gem "wdm", "~> 0.2.0", :platforms => [:mingw, :x64_mingw, :mswin]

gem "webrick", "~> 1.8"
