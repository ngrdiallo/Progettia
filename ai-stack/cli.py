from __future__ import annotations

from router_core import RouterError, default_router


def print_help() -> None:
    print("Comandi:")
    print("  /help                 Mostra aiuto")
    print("  /models               Elenca modelli locali")
    print("  /profiles             Elenca profili disponibili")
    print("  /profile <nome>       Imposta profilo comportamento")
    print("  /auto                 Modalita auto-routing")
    print("  /manual <model>       Forza modello")
    print("  /confirm on|off       Conferma azioni irreversibili")
    print("  /reload               Ricarica config")
    print("  /exit                 Esci")
    print("Suggerimento: puoi anche usare '# model: nome-modello' nella prima riga del prompt.")


def main() -> None:
    router = default_router()
    mode = "auto"
    manual_model = None
    profile = None
    confirm_irreversible = False

    print("=== Local Multi-Model Router CLI ===")
    print_help()

    while True:
        try:
            prompt = input("\nYou> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nUscita.")
            break

        if not prompt:
            continue

        if prompt == "/exit":
            break
        if prompt == "/help":
            print_help()
            continue
        if prompt == "/models":
            try:
                models = router.available_models()
                print("\nModelli locali:")
                for m in models:
                    print(f"- {m}")
            except Exception as e:
                print(f"Errore lettura modelli: {e}")
            continue
        if prompt == "/profiles":
            try:
                info = router.available_profiles()
                print(f"\nProfilo di default: {info.get('default_profile')}")
                print("Profili disponibili:")
                for p in info.get("profiles", []):
                    marker = " (attivo)" if p == profile else ""
                    print(f"- {p}{marker}")
            except Exception as e:
                print(f"Errore lettura profili: {e}")
            continue
        if prompt.startswith("/profile "):
            chosen = prompt.split(" ", 1)[1].strip()
            if not chosen:
                print("Specifica un profilo: /profile nome-profilo")
                continue
            profile = chosen
            print(f"Profilo attivo: {profile}")
            continue
        if prompt == "/auto":
            mode = "auto"
            manual_model = None
            print("Auto-routing attivo.")
            continue
        if prompt.startswith("/manual "):
            chosen = prompt.split(" ", 1)[1].strip()
            if not chosen:
                print("Specifica un modello: /manual nome-modello")
                continue
            mode = "manual"
            manual_model = chosen
            print(f"Manual override attivo: {manual_model}")
            continue
        if prompt == "/reload":
            router.reload_config()
            print("Config ricaricata.")
            continue
        if prompt.startswith("/confirm "):
            value = prompt.split(" ", 1)[1].strip().lower()
            if value not in {"on", "off"}:
                print("Uso: /confirm on|off")
                continue
            confirm_irreversible = value == "on"
            print(
                "Conferma azioni irreversibili: "
                + ("ATTIVA" if confirm_irreversible else "DISATTIVA")
            )
            continue

        try:
            result = router.generate(
                prompt=prompt,
                mode=mode,
                manual_model=manual_model,
                profile=profile,
                confirm_irreversible=confirm_irreversible,
            )
            print(
                f"\n[model] {result['model_used']} | [intent] {result['intent']} | "
                f"[mode] {result['mode']} | [profile] {result.get('profile', '-') }"
            )
            if result["fallback_used"]:
                print("[info] fallback usato")
            if result.get("output_sanitized"):
                print("[info] output ripulito da token di controllo")
            if result["errors"]:
                print("[trace] " + " | ".join(result["errors"]))
            print("\nAI> " + result["response"])
        except RouterError as e:
            print(f"Errore router: {e}")
        except Exception as e:
            print(f"Errore runtime: {e}")


if __name__ == "__main__":
    main()
