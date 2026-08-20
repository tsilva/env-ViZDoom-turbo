## PROJECT PURPOSE

`env-ViZDoom-turbo` provides high-throughput ViZDoom environments for reinforcement-learning research while retaining the upstream ViZDoom engine in a maintainable fork.

## PROJECT REQUIREMENTS

### Repository

- Use `env-ViZDoom-turbo` as the project and GitHub repository name, `env-vizdoom-turbo` as the Python distribution name, and `env_vizdoom_turbo` as the public Python import package; current project-owned identities must not use any former project, distribution, import, or command identifier.
- Maintain `env-ViZDoom-turbo` development on this repository’s `turbo` branch.
- Preserve the branded project README and repository-owned build/release workflow.
- Version `env-ViZDoom-turbo` releases as PEP 440 post releases whose base matches the pinned stable upstream ViZDoom version.
- Publish binary distributions only for Apple-silicon macOS and x86-64 Linux.
