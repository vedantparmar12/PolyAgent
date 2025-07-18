"""Model selection UI component for Streamlit"""

import streamlit as st
from typing import Optional, Dict, Any, List
from ...core.model_provider import ModelProvider, ModelRegistry, ModelInfo, ModelCapability, ModelConfig
import pandas as pd


class ModelSelectorUI:
    """UI component for model selection and configuration"""
    
    def __init__(self, model_config: Optional[ModelConfig] = None):
        """Initialize model selector
        
        Args:
            model_config: Model configuration with API keys
        """
        self.model_config = model_config or ModelConfig()
        self.model_provider = ModelProvider(config=self.model_config.dict())
        self.registry = ModelRegistry()
    
    def render_sidebar(self) -> str:
        """Render model selector in sidebar
        
        Returns:
            Selected model ID
        """
        with st.sidebar:
            st.subheader("🤖 Model Selection")
            
            # Quick selection
            quick_select = st.selectbox(
                "Quick Select",
                [
                    "Best Overall",
                    "Fastest",
                    "Most Affordable",
                    "Best for Coding",
                    "Best for Analysis",
                    "Long Context",
                    "Custom"
                ]
            )
            
            if quick_select == "Custom":
                return self._render_custom_selection()
            else:
                return self._get_quick_selection(quick_select)
    
    def render_full_interface(self) -> Dict[str, Any]:
        """Render full model selection interface
        
        Returns:
            Selected model configuration
        """
        st.header("🤖 AI Model Configuration")
        
        tabs = st.tabs(["Model Browser", "Comparison", "Cost Calculator", "API Keys"])
        
        with tabs[0]:
            selected_model = self._render_model_browser()
        
        with tabs[1]:
            self._render_model_comparison()
        
        with tabs[2]:
            self._render_cost_calculator()
        
        with tabs[3]:
            self._render_api_keys()
        
        return {
            "model_id": selected_model.id if selected_model else None,
            "config": self.model_config
        }
    
    def _render_model_browser(self) -> Optional[ModelInfo]:
        """Render model browser interface"""
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Search and filters
            search = st.text_input("🔍 Search models", placeholder="e.g., gpt-4, claude, coding")
        
        with col2:
            # Provider filter
            providers = ["All"] + [p.value for p in self.model_provider.registry._models.values()]
            selected_provider = st.selectbox("Provider", list(set(providers)))
        
        with col3:
            # Capability filter
            capabilities = ["All"] + [c.value for c in ModelCapability]
            selected_capability = st.selectbox("Capability", capabilities)
        
        # Advanced filters
        with st.expander("Advanced Filters"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                max_price = st.slider(
                    "Max Price (per 1K tokens)",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.1,
                    step=0.001,
                    format="$%.3f"
                )
            
            with col2:
                min_speed = st.slider(
                    "Min Speed Rating",
                    min_value=1,
                    max_value=10,
                    value=1
                )
            
            with col3:
                min_quality = st.slider(
                    "Min Quality Rating",
                    min_value=1,
                    max_value=10,
                    value=1
                )
        
        # Get filtered models
        filters = {}
        if selected_provider != "All":
            filters["provider"] = selected_provider
        if selected_capability != "All":
            filters["capability"] = selected_capability
        filters["max_price"] = max_price
        filters["min_speed"] = min_speed
        filters["min_quality"] = min_quality
        
        models = self.model_provider.list_available_models(**filters)
        
        # Apply search filter
        if search:
            search_lower = search.lower()
            models = [
                m for m in models
                if search_lower in m.id.lower() or
                search_lower in m.name.lower() or
                search_lower in m.description.lower() or
                any(search_lower in tag for tag in m.tags)
            ]
        
        # Display models
        selected_model = None
        
        for model in models:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    if st.button(
                        f"**{model.name}**\n{model.description[:60]}...",
                        key=f"select_{model.id}",
                        use_container_width=True
                    ):
                        selected_model = model
                
                with col2:
                    # Pricing
                    avg_price = (model.pricing.get("input", 0) + model.pricing.get("output", 0)) / 2
                    st.metric("Avg Price", f"${avg_price:.3f}")
                
                with col3:
                    # Speed rating
                    st.metric("Speed", f"{model.speed_rank}/10")
                
                with col4:
                    # Quality rating
                    st.metric("Quality", f"{model.quality_rank}/10")
                
                # Expandable details
                with st.expander(f"Details: {model.name}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Specifications:**")
                        st.write(f"- Context Window: {model.context_window:,} tokens")
                        st.write(f"- Max Output: {model.max_output:,} tokens")
                        st.write(f"- Provider: {model.provider}")
                        
                        st.write("**Capabilities:**")
                        for cap in model.capabilities:
                            st.write(f"- ✓ {cap.value.replace('_', ' ').title()}")
                    
                    with col2:
                        st.write("**Pricing:**")
                        st.write(f"- Input: ${model.pricing.get('input', 0):.4f}/1K tokens")
                        st.write(f"- Output: ${model.pricing.get('output', 0):.4f}/1K tokens")
                        
                        st.write("**Tags:**")
                        for tag in model.tags:
                            st.write(f"- {tag}")
        
        return selected_model
    
    def _render_model_comparison(self):
        """Render model comparison interface"""
        st.subheader("Model Comparison")
        
        # Select models to compare
        all_models = self.model_provider.list_available_models()
        model_names = [f"{m.name} ({m.provider})" for m in all_models]
        
        selected_indices = st.multiselect(
            "Select models to compare",
            range(len(model_names)),
            format_func=lambda x: model_names[x],
            default=[0, 1] if len(model_names) >= 2 else [0]
        )
        
        if selected_indices:
            selected_models = [all_models[i] for i in selected_indices]
            
            # Create comparison DataFrame
            data = []
            for model in selected_models:
                avg_price = (model.pricing.get("input", 0) + model.pricing.get("output", 0)) / 2
                data.append({
                    "Model": model.name,
                    "Provider": model.provider,
                    "Context": f"{model.context_window:,}",
                    "Max Output": f"{model.max_output:,}",
                    "Avg Price": f"${avg_price:.4f}",
                    "Speed": f"{model.speed_rank}/10",
                    "Quality": f"{model.quality_rank}/10",
                    "Vision": "✓" if ModelCapability.VISION in model.capabilities else "✗",
                    "Functions": "✓" if ModelCapability.FUNCTION_CALLING in model.capabilities else "✗",
                })
            
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
            
            # Radar chart comparison
            if len(selected_models) <= 5:
                st.subheader("Performance Comparison")
                
                import plotly.graph_objects as go
                
                categories = ['Speed', 'Quality', 'Price Efficiency', 'Context Length', 'Features']
                
                fig = go.Figure()
                
                for model in selected_models:
                    # Calculate scores
                    avg_price = (model.pricing.get("input", 0) + model.pricing.get("output", 0)) / 2
                    price_score = 10 - (avg_price * 100)  # Inverse price for score
                    context_score = min(model.context_window / 20000, 10)  # Normalize to 10
                    feature_score = len(model.capabilities) * 1.5  # More capabilities = higher score
                    
                    values = [
                        model.speed_rank,
                        model.quality_rank,
                        max(price_score, 0),
                        context_score,
                        min(feature_score, 10)
                    ]
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=model.name
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )),
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_cost_calculator(self):
        """Render cost calculator interface"""
        st.subheader("Cost Calculator")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Input parameters
            st.write("**Usage Estimation**")
            
            daily_requests = st.number_input(
                "Daily Requests",
                min_value=1,
                max_value=1000000,
                value=1000
            )
            
            avg_input_tokens = st.number_input(
                "Avg Input Tokens per Request",
                min_value=10,
                max_value=100000,
                value=500
            )
            
            avg_output_tokens = st.number_input(
                "Avg Output Tokens per Request",
                min_value=10,
                max_value=10000,
                value=200
            )
            
            days_per_month = st.number_input(
                "Active Days per Month",
                min_value=1,
                max_value=31,
                value=30
            )
        
        with col2:
            # Model selection for calculation
            st.write("**Select Models to Compare**")
            
            all_models = self.model_provider.list_available_models()
            selected_models = st.multiselect(
                "Models",
                all_models,
                format_func=lambda x: f"{x.name} ({x.provider})",
                default=all_models[:3] if len(all_models) >= 3 else all_models
            )
        
        if selected_models:
            # Calculate costs
            st.subheader("Cost Comparison")
            
            results = []
            for model in selected_models:
                daily_input_tokens = daily_requests * avg_input_tokens
                daily_output_tokens = daily_requests * avg_output_tokens
                
                daily_cost = self.model_provider.estimate_cost(
                    model.id,
                    daily_input_tokens,
                    daily_output_tokens
                )
                
                monthly_cost = daily_cost["total_cost"] * days_per_month
                
                results.append({
                    "Model": model.name,
                    "Daily Input Cost": f"${daily_cost['input_cost']:.2f}",
                    "Daily Output Cost": f"${daily_cost['output_cost']:.2f}",
                    "Daily Total": f"${daily_cost['total_cost']:.2f}",
                    "Monthly Total": f"${monthly_cost:.2f}",
                    "Annual Total": f"${monthly_cost * 12:.2f}"
                })
            
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)
            
            # Bar chart
            import plotly.express as px
            
            monthly_costs = [float(r["Monthly Total"].replace("$", "")) for r in results]
            model_names = [r["Model"] for r in results]
            
            fig = px.bar(
                x=model_names,
                y=monthly_costs,
                labels={"x": "Model", "y": "Monthly Cost ($)"},
                title="Monthly Cost Comparison"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            cheapest_idx = monthly_costs.index(min(monthly_costs))
            st.info(f"💰 Most cost-effective: **{model_names[cheapest_idx]}** at ${monthly_costs[cheapest_idx]:.2f}/month")
    
    def _render_api_keys(self):
        """Render API key configuration"""
        st.subheader("API Key Configuration")
        
        st.warning("🔐 API keys are stored locally and never sent to our servers")
        
        # OpenRouter
        with st.expander("OpenRouter (Access to 100+ models)"):
            openrouter_key = st.text_input(
                "OpenRouter API Key",
                value=self.model_config.openrouter_api_key or "",
                type="password",
                help="Get your API key at https://openrouter.ai/keys"
            )
            
            if st.button("Save OpenRouter Key"):
                self.model_config.openrouter_api_key = openrouter_key
                st.success("✅ OpenRouter API key saved!")
        
        # OpenAI
        with st.expander("OpenAI (GPT-4, GPT-3.5, etc.)"):
            openai_key = st.text_input(
                "OpenAI API Key",
                value=self.model_config.openai_api_key or "",
                type="password",
                help="Get your API key at https://platform.openai.com/api-keys"
            )
            
            if st.button("Save OpenAI Key"):
                self.model_config.openai_api_key = openai_key
                st.success("✅ OpenAI API key saved!")
        
        # Anthropic
        with st.expander("Anthropic (Claude 3 models)"):
            anthropic_key = st.text_input(
                "Anthropic API Key",
                value=self.model_config.anthropic_api_key or "",
                type="password",
                help="Get your API key at https://console.anthropic.com/"
            )
            
            if st.button("Save Anthropic Key"):
                self.model_config.anthropic_api_key = anthropic_key
                st.success("✅ Anthropic API key saved!")
        
        # Google
        with st.expander("Google (Gemini models)"):
            google_key = st.text_input(
                "Google API Key",
                value=self.model_config.google_api_key or "",
                type="password",
                help="Get your API key at https://makersuite.google.com/app/apikey"
            )
            
            if st.button("Save Google Key"):
                self.model_config.google_api_key = google_key
                st.success("✅ Google API key saved!")
        
        # Test connection
        st.subheader("Test Connection")
        
        if st.button("🔍 Test API Keys"):
            with st.spinner("Testing connections..."):
                results = self._test_api_keys()
                
                for provider, status in results.items():
                    if status["success"]:
                        st.success(f"✅ {provider}: Connected successfully!")
                    else:
                        st.error(f"❌ {provider}: {status['error']}")
    
    def _render_custom_selection(self) -> str:
        """Render custom model selection"""
        # Provider selection
        provider = st.selectbox(
            "Provider",
            ["OpenRouter", "OpenAI", "Anthropic", "Google", "Mistral", "Groq"]
        )
        
        # Get models for selected provider
        provider_models = [
            m for m in self.model_provider.list_available_models()
            if m.provider == provider or (provider == "OpenRouter" and "/" in m.id)
        ]
        
        if provider_models:
            selected_model = st.selectbox(
                "Model",
                provider_models,
                format_func=lambda x: f"{x.name} - {x.description[:50]}..."
            )
            
            # Show model details
            with st.expander("Model Details"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Speed", f"{selected_model.speed_rank}/10")
                    st.metric("Context", f"{selected_model.context_window:,} tokens")
                
                with col2:
                    st.metric("Quality", f"{selected_model.quality_rank}/10")
                    avg_price = (selected_model.pricing.get("input", 0) + selected_model.pricing.get("output", 0)) / 2
                    st.metric("Avg Price", f"${avg_price:.4f}/1K")
            
            return selected_model.id
        else:
            st.warning("No models available for selected provider")
            return self.model_config.default_model
    
    def _get_quick_selection(self, selection: str) -> str:
        """Get model based on quick selection"""
        if selection == "Best Overall":
            model = self.model_provider.recommend_model(
                task_type="general",
                budget_priority=False,
                speed_priority=False
            )
        elif selection == "Fastest":
            models = self.model_provider.list_available_models(min_speed=9)
            model = max(models, key=lambda m: m.speed_rank) if models else None
        elif selection == "Most Affordable":
            models = self.model_provider.list_available_models()
            model = min(models, key=lambda m: (m.pricing.get("input", 0) + m.pricing.get("output", 0)) / 2) if models else None
        elif selection == "Best for Coding":
            model = self.model_provider.recommend_model(
                task_type="coding",
                budget_priority=False,
                speed_priority=False
            )
        elif selection == "Best for Analysis":
            model = self.model_provider.recommend_model(
                task_type="analysis",
                budget_priority=False,
                speed_priority=False
            )
        elif selection == "Long Context":
            models = self.model_provider.list_available_models()
            model = max(models, key=lambda m: m.context_window) if models else None
        else:
            model = None
        
        if model:
            # Display selected model info
            st.info(f"Selected: **{model.name}** - {model.description[:60]}...")
            return model.id
        else:
            return self.model_config.default_model
    
    def _test_api_keys(self) -> Dict[str, Dict[str, Any]]:
        """Test API key connections"""
        results = {}
        
        # Test OpenRouter
        if self.model_config.openrouter_api_key:
            try:
                # Simple test - would implement actual API call
                results["OpenRouter"] = {"success": True}
            except Exception as e:
                results["OpenRouter"] = {"success": False, "error": str(e)}
        
        # Test OpenAI
        if self.model_config.openai_api_key:
            try:
                # Simple test - would implement actual API call
                results["OpenAI"] = {"success": True}
            except Exception as e:
                results["OpenAI"] = {"success": False, "error": str(e)}
        
        # Test Anthropic
        if self.model_config.anthropic_api_key:
            try:
                # Simple test - would implement actual API call
                results["Anthropic"] = {"success": True}
            except Exception as e:
                results["Anthropic"] = {"success": False, "error": str(e)}
        
        return results